import math
import functools
import torch
import cv2
import numpy as np
from utils import get_filled_mask
from visualization import save_images, save_optimization_gif
from typing import Dict, Any, Callable, Tuple
from rendering import Renderer
from score_sde.sde_lib import VESDE


def get_grad_texture(texture: torch.Tensor, grad_face: torch.Tensor, render_func: Callable) -> torch.Tensor:
    _, grad_texture = torch.autograd.functional.vjp(func=render_func, inputs=texture, v=grad_face, create_graph=False, strict=False)
    return grad_texture


def has_dark_color_due_to_background(gray: torch.Tensor, y: int, x: int, min_gray_intensity_in_texture: float, num_pixels_to_check_in_each_direction: int, image_size: int) -> bool:
    if gray[y, x] >= min_gray_intensity_in_texture:
        return False

    check_left = (x >= num_pixels_to_check_in_each_direction - 2)
    check_right = (x <= image_size - num_pixels_to_check_in_each_direction + 1)
    check_top = (y >= num_pixels_to_check_in_each_direction - 2)
    check_bottom = (y <= image_size - num_pixels_to_check_in_each_direction + 1)
    check_top_left = check_top and check_left
    check_top_right = check_top and check_right
    check_bottom_left = check_bottom and check_left
    check_bottom_right = check_bottom and check_right

    return (check_left and np.all([gray[y, x - i] < min_gray_intensity_in_texture for i in range(num_pixels_to_check_in_each_direction - 1) if x - i >= 0])) \
            or \
           (check_right and np.all([gray[y, x + i] < min_gray_intensity_in_texture for i in range(num_pixels_to_check_in_each_direction - 1) if x + i <= image_size - 1])) \
            or \
           (check_top and np.all([gray[y - i, x] < min_gray_intensity_in_texture for i in range(num_pixels_to_check_in_each_direction - 1) if y - i >= 0])) \
            or \
           (check_bottom and np.all([gray[y + i, x] < min_gray_intensity_in_texture for i in range(num_pixels_to_check_in_each_direction - 1) if y + i <= image_size - 1])) \
            or \
           (check_top_left and np.all([gray[y - i, x - i] < min_gray_intensity_in_texture for i in range(num_pixels_to_check_in_each_direction - 1) if y - i >= 0 and x - i >= 0])) \
            or \
           (check_top_right and np.all([gray[y - i, x + i] < min_gray_intensity_in_texture for i in range(num_pixels_to_check_in_each_direction - 1) if y - i >= 0 and x + i <= image_size - 1])) \
            or \
           (check_bottom_left and np.all([gray[y + i, x - i] < min_gray_intensity_in_texture for i in range(num_pixels_to_check_in_each_direction - 1) if y + i <= image_size - 1 and x - i >= 0])) \
            or \
           (check_bottom_right and np.all([gray[y + i, x + i] < min_gray_intensity_in_texture for i in range(num_pixels_to_check_in_each_direction - 1) if y + i <= image_size - 1 and x + i <= image_size - 1]))


def update_small_texture_using_face(cfg: Dict[str, Any], current_texture: torch.Tensor, pixel_uvs: torch.Tensor, current_optimized_face_mean: torch.Tensor, unfilled_mask: torch.Tensor, dark_pixel_allowance: bool) -> torch.Tensor:
    small_texture_size = cfg["small_texture_size"]
    batch_size = cfg["batch_size"]
    image_size = cfg["image_size"]
    min_gray_intensity_in_texture = cfg["min_gray_intensity_in_texture"]
    num_pixels_to_check_in_each_direction = cfg["num_pixels_to_check_in_each_direction"]

    current_optimized_face_mean_clamped = torch.clamp(current_optimized_face_mean, min=0.0, max=1.0)

    unfilled_mask_clone = unfilled_mask.clone()  # delete

    for sample_idx in range(batch_size):
        if not dark_pixel_allowance:
            gray = cv2.cvtColor(current_optimized_face_mean_clamped[sample_idx, :, :, :].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2GRAY)
        for y in range(image_size):
            for x in range(image_size):
                if unfilled_mask[sample_idx, 0, y, x] == 0:
                    continue
                
                # delete
                if (not dark_pixel_allowance) and (has_dark_color_due_to_background(gray, y, x, min_gray_intensity_in_texture, num_pixels_to_check_in_each_direction)):
                    unfilled_mask_clone[sample_idx, 0, y, x] = 1.0
                    unfilled_mask_clone[sample_idx, 1, y, x] = 0.0
                    unfilled_mask_clone[sample_idx, 2, y, x] = 0.0
                    continue

                u = math.floor((pixel_uvs[sample_idx, y, x, 0] + 1) * small_texture_size / 2)
                v = small_texture_size - 1 - math.floor(small_texture_size * (pixel_uvs[sample_idx, y, x, 1] + 1) / 2)

                current_texture[sample_idx, :, v, u] = current_optimized_face_mean_clamped[sample_idx, :, y, x]
    
    save_images(cfg=cfg, _images=unfilled_mask_clone, image_type="unfilled_mask_clone", iteration=0, dark_pixel_allowance=dark_pixel_allowance, elev=0.0, azimuth=0.0)

    return current_texture


def update_small_texture_using_large_texture(cfg: Dict[str, Any], small_texture: torch.Tensor, large_texture: torch.Tensor, pixel_uvs: torch.Tensor, dark_pixel_allowance: bool) -> torch.Tensor:
    small_texture_size = cfg["small_texture_size"]
    large_texture_size = cfg["large_texture_size"]
    min_gray_intensity_in_texture = cfg["min_gray_intensity_in_texture"]

    large_texture_clamped = torch.clamp(large_texture, min=0.0, max=1.0)
    for sample_idx in range(pixel_uvs.shape[0]):
        if not dark_pixel_allowance:
            gray = cv2.cvtColor(large_texture_clamped[sample_idx, :, :, :].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2GRAY)
        for y in range(pixel_uvs.shape[1]):
            for x in range(pixel_uvs.shape[2]):
                u_small = math.floor((pixel_uvs[sample_idx, y, x, 0] + 1) * small_texture_size / 2)
                v_small = small_texture_size - 1 - math.floor(small_texture_size * (pixel_uvs[sample_idx, y, x, 1] + 1) / 2)
                u_large = math.floor((pixel_uvs[sample_idx, y, x, 0] + 1) * large_texture_size / 2)
                v_large = large_texture_size - 1 - math.floor(large_texture_size * (pixel_uvs[sample_idx, y, x, 1] + 1) / 2)
                if ((u_small == 0) and (v_small == small_texture_size - 1)):
                    continue
                elif not dark_pixel_allowance and gray[v_large, u_large] < min_gray_intensity_in_texture:
                    continue
                else:
                    small_texture[sample_idx, :, v_small, u_small] = large_texture_clamped[sample_idx, :, v_large, u_large]
    return small_texture


def update_large_texture_by_upsampling_small_texture(cfg: Dict[str, Any], small_texture: torch.Tensor, initial_large_texture: torch.Tensor, initial_filled_texture_mask: torch.Tensor) -> torch.Tensor:
    large_texture_size = cfg["large_texture_size"]
    upsampled_small_texture = torch.nn.functional.interpolate(input=small_texture, size=large_texture_size, mode='nearest')
    return torch.where(initial_filled_texture_mask, initial_large_texture, upsampled_small_texture)


def run_single_view_optimization_in_texture_space(cfg: Dict[str, Any], current_small_texture: torch.Tensor, current_large_texture: torch.Tensor, initial_large_texture: torch.Tensor, initial_filled_texture_mask: torch.Tensor, sde: VESDE, timesteps: torch.Tensor, elev: float, azimuth: float, quaternion: torch.Tensor, dark_pixel_allowance: bool, renderer: Renderer, target_background: torch.Tensor, score_fn: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = cfg["batch_size"]
    device = cfg["device"]
    num_corrector_steps = cfg["num_corrector_steps"]
    snr = cfg["snr"]
    save_images_kwargs = {"elev": elev, "azimuth": azimuth, "dark_pixel_allowance": dark_pixel_allowance, "cfg": cfg}

    current_filled_texture_mask = get_filled_mask(cfg=cfg, image=current_large_texture)
    initial_filled_texture = current_large_texture * current_filled_texture_mask
    current_unfilled_texture = torch.randn_like(current_large_texture) * sde.sigma_max * (1 - 1 * current_filled_texture_mask)

    pixel_uvs, background_mask = renderer.prerender(quaternion=quaternion)
    initial_face = renderer.render(texture=current_small_texture, background=target_background, pixel_uvs=pixel_uvs, background_mask=background_mask)

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

    current_small_texture = update_small_texture_using_large_texture(cfg=cfg, pixel_uvs=pixel_uvs, small_texture=current_small_texture, large_texture=current_large_texture, dark_pixel_allowance=dark_pixel_allowance)
    current_large_texture = update_large_texture_by_upsampling_small_texture(cfg=cfg, small_texture=current_small_texture, initial_large_texture=initial_large_texture, initial_filled_texture_mask=initial_filled_texture_mask)
    final_face = renderer.render(texture=current_large_texture, background=target_background, pixel_uvs=pixel_uvs, background_mask=background_mask)
    save_images(_images=current_small_texture, image_type="final_small_texture", iteration=0, **save_images_kwargs)
    save_images(_images=current_large_texture, image_type="final_large_texture", iteration=0, **save_images_kwargs)
    save_images(_images=final_face, image_type="final_face", iteration=0, **save_images_kwargs)
    save_optimization_gif(**save_images_kwargs)

    return current_small_texture, current_large_texture


def run_single_view_optimization_in_image_space(cfg: Dict[str, Any], elev: float, azimuth: float, quaternion: torch.Tensor, dark_pixel_allowance: bool, renderer: Renderer, target_background: torch.Tensor, sde: VESDE, current_small_texture: torch.Tensor, timesteps: torch.Tensor, score_model: torch.nn.DataParallel, predictor_inpaint_update_fn: Callable, corrector_inpaint_update_fn: Callable) -> torch.Tensor:
    save_images_kwargs = {"cfg": cfg, "elev": elev, "azimuth": azimuth, "dark_pixel_allowance": dark_pixel_allowance}

    pixel_uvs, background_mask = renderer.prerender(quaternion=quaternion)
    current_unoptimized_face = renderer.render(texture=current_small_texture, background=target_background, pixel_uvs=pixel_uvs, background_mask=background_mask)
    unfilled_mask = (current_unoptimized_face.sum(axis=1, keepdim=True) == 3).repeat(repeats=[1, 3, 1, 1]) * (1 - 1 * background_mask)  # white pixels are unfilled
    filled_mask = 1 - unfilled_mask # other pixels are filled

    current_optimized_face = current_unoptimized_face * filled_mask + sde.prior_sampling(current_unoptimized_face.shape).to(current_unoptimized_face.device) * unfilled_mask

    save_images(_images=unfilled_mask, image_type="unfilled_mask", iteration=0, **save_images_kwargs)
    save_images(_images=filled_mask, image_type="filled_mask", iteration=0, **save_images_kwargs)

    save_images(_images=current_small_texture, image_type="initial_texture", iteration=0, **save_images_kwargs)
    save_images(_images=current_unoptimized_face, image_type="initial_face", iteration=0, **save_images_kwargs)

    for i in range(sde.N):
        t = timesteps[i]
        current_optimized_face, current_optimized_face_mean = predictor_inpaint_update_fn(score_model, current_unoptimized_face, filled_mask, current_optimized_face, t)
        current_optimized_face, current_optimized_face_mean = corrector_inpaint_update_fn(score_model, current_unoptimized_face, filled_mask, current_optimized_face, t)
        if i % 10 == 0:
            save_images(_images=current_optimized_face_mean, image_type="face", iteration=i, **save_images_kwargs)

    current_small_texture = update_small_texture_using_face(cfg=cfg, current_texture=current_small_texture, pixel_uvs=pixel_uvs.clone(), current_optimized_face_mean=current_optimized_face_mean.clone(), unfilled_mask=unfilled_mask, dark_pixel_allowance=dark_pixel_allowance)
    current_optimized_face = renderer.render(texture=current_small_texture, background=torch.zeros_like(target_background), pixel_uvs=pixel_uvs, background_mask=background_mask)

    save_images(_images=current_small_texture, image_type="final_texture", iteration=0, **save_images_kwargs)
    save_images(_images=current_optimized_face, image_type="final_face", iteration=0, **save_images_kwargs)

    save_optimization_gif(**save_images_kwargs)

    return current_small_texture
