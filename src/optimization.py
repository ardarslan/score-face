import math
import itertools
import functools
import torch
import cv2
import numpy as np
from tqdm import tqdm
from utils import get_filled_mask, get_initial_textures
from model_utils import get_reverse_diffusion_predictor_inpaint_update_fn, get_langevin_corrector_inpaint_update_fn
from visualization import save_images, save_optimization_gif, save_outputs
from typing import Dict, Any, Callable, Tuple, List, OrderedDict
from rendering import Renderer
from score_sde.sde_lib import VESDE


def get_grad_texture(texture: torch.Tensor, grad_face: torch.Tensor, render_func: Callable) -> torch.Tensor:
    _, grad_texture = torch.autograd.functional.vjp(func=render_func, inputs=texture, v=grad_face, create_graph=False, strict=False)
    return grad_texture


def whiten_large_texture(cfg: Dict[str, Any], pixel_uvs: torch.Tensor, pixels_to_whiten: List[Tuple[int, int, int]], large_texture: torch.Tensor) -> torch.Tensor:
    device = cfg["device"]
    large_texture_size = cfg["large_texture_size"]

    for sample_idx, y, x in pixels_to_whiten:
        u_large = math.floor((pixel_uvs[sample_idx, y, x, 0] + 1) * large_texture_size / 2)
        v_large = large_texture_size - 1 - math.floor(large_texture_size * (pixel_uvs[sample_idx, y, x, 1] + 1) / 2)
        large_texture[sample_idx, :, v_large, u_large] = torch.tensor([1.0, 1.0, 1.0], device=device)
    
    return large_texture


def whiten_face(cfg: Dict[str, Any], face: torch.Tensor, pixels_to_whiten: List[Tuple[int, int, int]]) -> torch.Tensor:
    device = cfg["device"]

    for sample_idx, y, x in pixels_to_whiten:
        face[sample_idx, :, y, x] = torch.tensor([1.0, 1.0, 1.0], device=device)

    return face


def get_pixels_to_whiten(cfg: Dict[str, Any], face: torch.Tensor, unfilled_face_mask: torch.Tensor, background_mask: torch.Tensor) -> bool:
    batch_size = cfg["batch_size"]
    image_size = cfg["image_size"]
    device = cfg["device"]
    min_gray_intensity_in_texture = cfg["min_gray_intensity_in_texture"]

    def is_any_neighbor_background(sample_idx, y, x, background_mask, image_size):
        return (x > 0 and background_mask[sample_idx, 0, y, x - 1]) or \
               (x < image_size - 1 and background_mask[sample_idx, 0, y, x + 1]) or \
               (y > 0 and background_mask[sample_idx, 0, y - 1, x]) or \
               (y < image_size - 1 and background_mask[sample_idx, 0, y + 1, x])
    
    def is_dark(sample_idx, y, x, dark_mask):
        return dark_mask[sample_idx, y, x]
    
    def is_unfilled(sample_idx, y, x, unfilled_face_mask):
        return unfilled_face_mask[sample_idx, 0, y, x]
    
    def is_unexplored(sample_idx, y, x, explored_pixels):
        return (sample_idx, y, x) not in explored_pixels
    
    def get_unexplored_unfilled_dark_neighbors(sample_idx, y, x, image_size, explored_pixels):
        results = set()

        if x > 0 and is_dark(sample_idx, y, x - 1, dark_mask) and is_unexplored(sample_idx, y, x - 1, explored_pixels) and is_unfilled(sample_idx, y, x - 1, unfilled_face_mask):
            results.add((sample_idx, y, x - 1))

        if x < image_size - 1 and is_dark(sample_idx, y, x + 1, dark_mask) and is_unexplored(sample_idx, y, x + 1, explored_pixels) and is_unfilled(sample_idx, y, x + 1, unfilled_face_mask):
            results.add((sample_idx, y, x + 1))

        if y > 0 and is_dark(sample_idx, y - 1, x, dark_mask) and is_unexplored(sample_idx, y - 1, x, explored_pixels) and is_unfilled(sample_idx, y - 1, x, unfilled_face_mask):
            results.add((sample_idx, y - 1, x))

        if y < image_size - 1 and is_dark(sample_idx, y + 1, x, dark_mask) and is_unexplored(sample_idx, y + 1, x, explored_pixels) and is_unfilled(sample_idx, y + 1, x, unfilled_face_mask):
            results.add((sample_idx, y + 1, x))

        return results

    face_clamped = torch.clamp(face, min=0.0, max=1.0)

    dark_mask = []
    for sample_idx in range(batch_size):
        dark_mask.append((torch.tensor(cv2.cvtColor(face_clamped[sample_idx].permute(1, 2, 0).cpu().numpy().astype(np.float32), cv2.COLOR_RGB2GRAY), device=device) < min_gray_intensity_in_texture).unsqueeze(0))
    dark_mask = torch.concat(dark_mask)

    pixels_to_explore = set()
    pixels_to_whiten = []

    for sample_idx in range(batch_size):
        for y in range(image_size):
            for x in range(image_size):
                if is_unfilled(sample_idx, y, x, unfilled_face_mask) and is_any_neighbor_background(sample_idx, y, x, background_mask, image_size) and is_dark(sample_idx, y, x, dark_mask):
                    pixels_to_explore.add((sample_idx, y, x))
                    pixels_to_whiten.append((sample_idx, y, x))

    explored_pixels = set()
    while len(pixels_to_explore) > 0:
        pixel_to_explore = pixels_to_explore.pop()
        sample_idx, y, x = pixel_to_explore
        unexplored_unfilled_dark_neighbors = get_unexplored_unfilled_dark_neighbors(sample_idx, y, x, image_size, explored_pixels)
        pixels_to_explore = pixels_to_explore.union(unexplored_unfilled_dark_neighbors)
        pixels_to_whiten.extend(unexplored_unfilled_dark_neighbors)
        explored_pixels.add(pixel_to_explore)
    
    return pixels_to_whiten


def update_small_texture_using_face(cfg: Dict[str, Any], current_texture: torch.Tensor, pixel_uvs: torch.Tensor, current_optimized_face_mean: torch.Tensor, unfilled_face_mask: torch.Tensor) -> torch.Tensor:
    small_texture_size = cfg["small_texture_size"]
    batch_size = cfg["batch_size"]
    image_size = cfg["image_size"]

    current_optimized_face_mean_clamped = torch.clamp(current_optimized_face_mean, min=0.0, max=1.0)

    for sample_idx in range(batch_size):
        for y in range(image_size):
            for x in range(image_size):
                if unfilled_face_mask[sample_idx, 0, y, x] == 0:
                    continue

                u = math.floor((pixel_uvs[sample_idx, y, x, 0] + 1) * small_texture_size / 2)
                v = small_texture_size - 1 - math.floor(small_texture_size * (pixel_uvs[sample_idx, y, x, 1] + 1) / 2)

                current_texture[sample_idx, :, v, u] = current_optimized_face_mean_clamped[sample_idx, :, y, x]

    return current_texture


def update_small_texture_using_large_texture(cfg: Dict[str, Any], small_texture: torch.Tensor, large_texture: torch.Tensor, pixel_uvs: torch.Tensor) -> torch.Tensor:
    small_texture_size = cfg["small_texture_size"]
    large_texture_size = cfg["large_texture_size"]

    large_texture_clamped = torch.clamp(large_texture, min=0.0, max=1.0)
    for sample_idx in range(pixel_uvs.shape[0]):
        for y in range(pixel_uvs.shape[1]):
            for x in range(pixel_uvs.shape[2]):
                u_small = math.floor((pixel_uvs[sample_idx, y, x, 0] + 1) * small_texture_size / 2)
                v_small = small_texture_size - 1 - math.floor(small_texture_size * (pixel_uvs[sample_idx, y, x, 1] + 1) / 2)
                u_large = math.floor((pixel_uvs[sample_idx, y, x, 0] + 1) * large_texture_size / 2)
                v_large = large_texture_size - 1 - math.floor(large_texture_size * (pixel_uvs[sample_idx, y, x, 1] + 1) / 2)
                if ((u_small == 0) and (v_small == small_texture_size - 1)):
                    continue
                else:
                    small_texture[sample_idx, :, v_small, u_small] = large_texture_clamped[sample_idx, :, v_large, u_large]
    return small_texture


def update_large_texture_by_upsampling_small_texture(cfg: Dict[str, Any], small_texture: torch.Tensor, initial_large_texture: torch.Tensor, initial_filled_texture_mask: torch.Tensor) -> torch.Tensor:
    large_texture_size = cfg["large_texture_size"]
    upsampled_small_texture = torch.nn.functional.interpolate(input=small_texture, size=large_texture_size, mode='nearest')
    return torch.where(initial_filled_texture_mask, initial_large_texture, upsampled_small_texture)


def run_single_view_optimization_in_texture_space(cfg: Dict[str, Any], current_small_texture: torch.Tensor, current_large_texture: torch.Tensor, initial_large_texture: torch.Tensor, initial_filled_texture_mask: torch.Tensor, sde: VESDE, timesteps: torch.Tensor, elev: float, azimuth: float, dark_pixel_allowed: bool, renderer: Renderer, target_background: torch.Tensor, score_fn: Callable, pixel_uvs: torch.Tensor, background_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = cfg["batch_size"]
    device = cfg["device"]
    num_corrector_steps = cfg["num_corrector_steps"]
    snr = cfg["snr"]
    num_channels = cfg["num_channels"]

    save_images_kwargs = {"elev": elev, "azimuth": azimuth, "dark_pixel_allowed": dark_pixel_allowed, "cfg": cfg}

    current_filled_texture_mask = get_filled_mask(cfg=cfg, image=current_large_texture)
    initial_filled_texture = current_large_texture * current_filled_texture_mask
    current_unfilled_texture = torch.randn_like(current_large_texture) * sde.sigma_max * (1 - 1 * current_filled_texture_mask)

    initial_face = renderer.render(texture=current_large_texture, background=target_background, pixel_uvs=pixel_uvs, background_mask=background_mask)
    unfilled_face_mask = (initial_face.sum(axis=1, keepdim=True) == num_channels).repeat(repeats=[1, num_channels, 1, 1]) * (1 - 1 * background_mask)

    save_images(_images=current_small_texture, image_type="initial_small_texture", iteration=0, **save_images_kwargs)
    save_images(_images=current_large_texture, image_type="initial_large_texture", iteration=0, **save_images_kwargs)
    save_images(_images=initial_face, image_type="initial_face", iteration=0, **save_images_kwargs)

    for i in range(sde.N):
        t = timesteps[i]
        vec_t = torch.ones(batch_size, device=device) * t
        alpha = torch.ones_like(vec_t)

        background_mean, background_std = sde.marginal_prob(target_background, vec_t)
        background = background_mean + torch.randn_like(background_mean) * background_std[:, None, None, None]

        current_filled_texture_mean, current_filled_texture_std = sde.marginal_prob(initial_filled_texture, vec_t)
        current_filled_texture = current_filled_texture_mean + torch.randn_like(current_filled_texture_mean) * current_filled_texture_std[:, None, None, None]

        if i == 0:
            current_large_texture = torch.where(current_filled_texture_mask, current_filled_texture, current_unfilled_texture)
            current_face = renderer.render(texture=current_large_texture, background=background, pixel_uvs=pixel_uvs, background_mask=background_mask)
            render_func = functools.partial(renderer.render, background=background, pixel_uvs=pixel_uvs, background_mask=background_mask)

        for _ in range(num_corrector_steps):
            grad_face = score_fn(current_face, vec_t)
            grad_texture = get_grad_texture(texture=current_large_texture, grad_face=grad_face, render_func=render_func)
            grad_texture_norm = torch.norm(grad_texture.reshape(grad_texture.shape[0], -1), dim=-1).mean()
            noise_texture = torch.randn_like(grad_texture)
            noise_texture_norm = torch.norm(noise_texture.reshape(noise_texture.shape[0], -1), dim=-1).mean()
            step_size_texture = (snr * noise_texture_norm / grad_texture_norm) ** 2 * 2 * alpha
            current_unfilled_texture_mean = current_unfilled_texture + step_size_texture[:, None, None, None] * grad_texture
            current_unfilled_texture = current_unfilled_texture_mean + torch.sqrt(step_size_texture * 2)[:, None, None, None] * noise_texture
            current_large_texture = torch.where(current_filled_texture_mask, current_filled_texture, current_unfilled_texture)
            current_face = renderer.render(texture=current_large_texture,
                                            background=background,
                                            pixel_uvs=pixel_uvs,
                                            background_mask=background_mask)
            render_func = functools.partial(renderer.render, background=background, pixel_uvs=pixel_uvs, background_mask=background_mask)

        if i % 10 == 0:
            save_images(_images=current_face, image_type="face", iteration=i, **save_images_kwargs)

    if not dark_pixel_allowed:
        pixels_to_whiten = get_pixels_to_whiten(cfg=cfg, face=current_face, unfilled_face_mask=unfilled_face_mask, background_mask=background_mask)
        current_large_texture = whiten_large_texture(cfg=cfg, pixel_uvs=pixel_uvs, pixels_to_whiten=pixels_to_whiten, large_texture=current_large_texture)

    current_small_texture = update_small_texture_using_large_texture(cfg=cfg, pixel_uvs=pixel_uvs, small_texture=current_small_texture, large_texture=current_large_texture)
    current_large_texture = update_large_texture_by_upsampling_small_texture(cfg=cfg, small_texture=current_small_texture, initial_large_texture=initial_large_texture, initial_filled_texture_mask=initial_filled_texture_mask)
    final_face = renderer.render(texture=current_large_texture, background=target_background, pixel_uvs=pixel_uvs, background_mask=background_mask)
    save_images(_images=current_small_texture, image_type="final_small_texture", iteration=0, **save_images_kwargs)
    save_images(_images=current_large_texture, image_type="final_large_texture", iteration=0, **save_images_kwargs)
    save_images(_images=final_face, image_type="final_face", iteration=0, **save_images_kwargs)
    save_optimization_gif(**save_images_kwargs)

    return current_small_texture, current_large_texture


def run_single_view_optimization_in_image_space(cfg: Dict[str, Any], elev: float, azimuth: float, dark_pixel_allowed: bool, renderer: Renderer, target_background: torch.Tensor, sde: VESDE, current_small_texture: torch.Tensor, timesteps: torch.Tensor, score_model: torch.nn.DataParallel, predictor_inpaint_update_fn: Callable, corrector_inpaint_update_fn: Callable, pixel_uvs: torch.Tensor, background_mask: torch.Tensor) -> torch.Tensor:
    num_channels = cfg["num_channels"]
    save_images_kwargs = {"cfg": cfg, "elev": elev, "azimuth": azimuth, "dark_pixel_allowed": dark_pixel_allowed}

    current_unoptimized_face = renderer.render(texture=current_small_texture, background=target_background, pixel_uvs=pixel_uvs, background_mask=background_mask)
    unfilled_face_mask = (current_unoptimized_face.sum(axis=1, keepdim=True) == num_channels).repeat(repeats=[1, num_channels, 1, 1]) * (1 - 1 * background_mask)  # white pixels are unfilled
    filled_face_mask = 1 - unfilled_face_mask # other pixels are filled

    current_optimized_face = current_unoptimized_face * filled_face_mask + sde.prior_sampling(current_unoptimized_face.shape).to(current_unoptimized_face.device) * unfilled_face_mask

    save_images(_images=unfilled_face_mask, image_type="unfilled_face_mask", iteration=0, **save_images_kwargs)
    save_images(_images=filled_face_mask, image_type="filled_face_mask", iteration=0, **save_images_kwargs)

    save_images(_images=current_small_texture, image_type="initial_texture", iteration=0, **save_images_kwargs)
    save_images(_images=current_unoptimized_face, image_type="initial_face", iteration=0, **save_images_kwargs)

    for i in range(sde.N):
        t = timesteps[i]
        current_optimized_face, current_optimized_face_mean = predictor_inpaint_update_fn(score_model, current_unoptimized_face, filled_face_mask, current_optimized_face, t)
        current_optimized_face, current_optimized_face_mean = corrector_inpaint_update_fn(score_model, current_unoptimized_face, filled_face_mask, current_optimized_face, t)
        if i % 10 == 0:
            save_images(_images=current_optimized_face_mean, image_type="face", iteration=i, **save_images_kwargs)

    if not dark_pixel_allowed:
        pixels_to_whiten = get_pixels_to_whiten(cfg=cfg, face=current_optimized_face_mean, unfilled_face_mask=unfilled_face_mask, background_mask=background_mask)
        current_optimized_face_mean = whiten_face(cfg=cfg, face=current_optimized_face_mean, pixels_to_whiten=pixels_to_whiten)

    current_small_texture = update_small_texture_using_face(cfg=cfg, current_texture=current_small_texture, pixel_uvs=pixel_uvs.clone(), current_optimized_face_mean=current_optimized_face_mean.clone(), unfilled_face_mask=unfilled_face_mask)
    current_optimized_face = renderer.render(texture=current_small_texture, background=torch.zeros_like(target_background), pixel_uvs=pixel_uvs, background_mask=background_mask)

    save_images(_images=current_small_texture, image_type="final_texture", iteration=0, **save_images_kwargs)
    save_images(_images=current_optimized_face, image_type="final_face", iteration=0, **save_images_kwargs)

    save_optimization_gif(**save_images_kwargs)

    return current_small_texture


def run_multi_view_optimization_in_texture_space(cfg: Dict[str, Any], sde: VESDE, timesteps: torch.Tensor, renderer: Renderer, target_background: torch.Tensor, score_fn: Callable, dark_pixel_alloweds: List[bool], elevs: List[float], azimuths: List[float], ordered_prerender_results: OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]]) -> None:
    initial_small_texture, initial_large_texture = get_initial_textures(cfg=cfg)
    current_small_texture = initial_small_texture.clone()
    current_large_texture = initial_large_texture.clone()
    initial_filled_texture_mask = get_filled_mask(cfg=cfg, image=initial_large_texture)
    with torch.no_grad():
        for dark_pixel_allowed, (elev_azimuth, (pixel_uvs, background_mask)) in tqdm(list(itertools.product(dark_pixel_alloweds, list(ordered_prerender_results.items())))):
            elev_azimuth_splitted = elev_azimuth.split("_")
            elev, azimuth = elev_azimuth_splitted[0], elev_azimuth_splitted[1]
            current_small_texture, current_large_texture = run_single_view_optimization_in_texture_space(cfg=cfg, current_small_texture=current_small_texture, current_large_texture=current_large_texture, initial_large_texture=initial_large_texture, initial_filled_texture_mask=initial_filled_texture_mask, sde=sde, timesteps=timesteps, elev=elev, azimuth=azimuth, dark_pixel_allowed=dark_pixel_allowed, renderer=renderer, target_background=target_background, score_fn=score_fn, pixel_uvs=pixel_uvs, background_mask=background_mask)
        save_outputs(cfg=cfg, final_texture=current_large_texture, target_background=torch.zeros_like(target_background), renderer=renderer, elevs=elevs, azimuths=azimuths, ordered_prerender_results=ordered_prerender_results)


def run_multi_view_optimization_in_image_space(cfg: Dict[str, Any], sde: VESDE, timesteps: torch.Tensor, renderer: Renderer, target_background: torch.Tensor, dark_pixel_alloweds: List[bool], elevs: List[float], azimuths: List[float], score_model: VESDE, ordered_prerender_results: OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]]) -> None:
    initial_small_texture, _ = get_initial_textures(cfg=cfg)
    current_small_texture = initial_small_texture.clone()
    predictor_inpaint_update_fn = get_reverse_diffusion_predictor_inpaint_update_fn(sde=sde)
    corrector_inpaint_update_fn = get_langevin_corrector_inpaint_update_fn(cfg=cfg, sde=sde)
    with torch.no_grad():
        for dark_pixel_allowed, (elev_azimuth, (pixel_uvs, background_mask)) in tqdm(list(itertools.product(dark_pixel_alloweds, list(ordered_prerender_results.items())))):
            elev_azimuth_splitted = elev_azimuth.split("_")
            elev, azimuth = elev_azimuth_splitted[0], elev_azimuth_splitted[1]
            current_small_texture = run_single_view_optimization_in_image_space(cfg=cfg, elev=elev, azimuth=azimuth, dark_pixel_allowed=dark_pixel_allowed, renderer=renderer, target_background=target_background, sde=sde, current_small_texture=current_small_texture, timesteps=timesteps, score_model=score_model, predictor_inpaint_update_fn=predictor_inpaint_update_fn, corrector_inpaint_update_fn=corrector_inpaint_update_fn, pixel_uvs=pixel_uvs, background_mask=background_mask)
        save_outputs(cfg=cfg, final_texture=current_small_texture, target_background=torch.zeros_like(target_background), renderer=renderer, elevs=elevs, azimuths=azimuths, ordered_prerender_results=ordered_prerender_results)
