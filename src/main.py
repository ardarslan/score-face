from utils import set_experiment_name, save_cfg, set_seeds, get_grad_texture, get_target_background, set_3dmm_result_paths, get_initial_textures, get_current_large_texture, get_current_small_texture, get_filled_mask
from model_utils import get_sde, get_score_model, get_score_fn
from visualization import save_images, save_outputs
from renderer import Renderer

import itertools
import functools
import torch
from tqdm import tqdm

cfg = {
    "seed": 42,
    "batch_size": 1,
    "image_size": 256,
    "large_texture_size": 2048,
    "small_texture_size": 256,
    "saved_image_size": 256,
    "num_channels": 3,
    "image_save_dir": "../results",
    "checkpoint_path": "../assets/checkpoint_48.pth",
    "device": torch.device("cuda:0"),
    "num_corrector_steps": 6,
    "sde_N": 1000,
    "snr": 0.015,
    "subject_id": 33673,
    "3dmm": "tf_flame",
    "camera_distance": 1.2
}
set_experiment_name(cfg=cfg)
set_3dmm_result_paths(cfg=cfg)
save_cfg(cfg=cfg)
set_seeds(seed=cfg["seed"])

sde = get_sde(sde_N=cfg["sde_N"])
score_model = get_score_model(checkpoint_path=cfg["checkpoint_path"], batch_size=cfg["batch_size"], device=cfg["device"])
score_fn = get_score_fn(score_model=score_model, sde=sde)

renderer = Renderer(image_size=cfg["image_size"], camera_distance=cfg["camera_distance"], device=cfg["device"])

target_background = get_target_background(batch_size=cfg["batch_size"], num_channels=cfg["num_channels"], image_size=cfg["image_size"], device=cfg["device"])
initial_small_texture, initial_large_texture = get_initial_textures(texture_path=cfg["texture_path"], _3dmm=cfg["3dmm"], small_texture_size=cfg["small_texture_size"], large_texture_size=cfg["large_texture_size"], device=cfg["device"])
initial_filled_mask = get_filled_mask(texture=initial_large_texture, num_channels=cfg["num_channels"])
current_small_texture = initial_small_texture.clone()
current_large_texture = initial_large_texture.clone()

timesteps = torch.linspace(sde.T, 1e-5, sde.N)

elevs = [0.0, 10.0, -10.0, 20.0, -20.0]
azimuths = [0.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0, 40.0, -40.0]
loop_tuples = list(itertools.product(elevs, azimuths))

with torch.no_grad():
    for elev, azimuth in tqdm(loop_tuples):
        save_images_kwargs = {"elev": elev, "azimuth": azimuth, "image_save_dir": cfg["image_save_dir"], "experiment_name": cfg["experiment_name"], "saved_image_size": cfg["saved_image_size"]}
        
        current_filled_mask = get_filled_mask(texture=current_large_texture, num_channels=cfg["num_channels"])
        initial_filled_texture = current_large_texture * current_filled_mask
        current_unfilled_texture = torch.randn_like(current_large_texture) * sde.sigma_max * (1 - 1 * current_filled_mask)

        pixel_uvs, background_mask = renderer.prerender(obj_path=cfg["obj_path"], axis_angle_path=cfg["axis_angle_path"], texture=current_large_texture, elev=elev, azimuth=azimuth)
        initial_face = renderer.render(texture=current_small_texture, background=target_background, pixel_uvs=pixel_uvs, background_mask=background_mask)

        save_images(_images=current_small_texture, image_type="initial_small_texture", iteration=0, **save_images_kwargs)
        save_images(_images=current_large_texture, image_type="initial_large_texture", iteration=0, **save_images_kwargs)
        save_images(_images=initial_face, image_type="initial_face", iteration=0, **save_images_kwargs)

        for i in range(sde.N):
            t = timesteps[i]
            vec_t = torch.ones(cfg["batch_size"], device=cfg["device"]) * t
            alpha = torch.ones_like(vec_t)

            background_mean, background_std = sde.marginal_prob(target_background, vec_t)
            background = background_mean + torch.randn_like(background_mean) * background_std[:, None, None, None]

            current_filled_texture_mean, current_filled_texture_std = sde.marginal_prob(initial_filled_texture, vec_t)
            current_filled_texture = current_filled_texture_mean + torch.randn_like(current_filled_texture_mean) * current_filled_texture_std[:, None, None, None]

            if i == 0:
                current_large_texture = torch.where(current_filled_mask, current_filled_texture, current_unfilled_texture)
                current_face = renderer.render(texture=current_large_texture, background=background, pixel_uvs=pixel_uvs, background_mask=background_mask)
                render_func = functools.partial(renderer.render, background=background, pixel_uvs=pixel_uvs, background_mask=background_mask)

            for j in range(cfg["num_corrector_steps"]):
                grad_face = score_fn(current_face, vec_t)
                grad_texture = get_grad_texture(texture=current_large_texture, grad_face=grad_face, render_func=render_func)
                grad_texture_norm = torch.norm(grad_texture.reshape(grad_texture.shape[0], -1), dim=-1).mean()
                noise_texture = torch.randn_like(grad_texture)
                noise_texture_norm = torch.norm(noise_texture.reshape(noise_texture.shape[0], -1), dim=-1).mean()
                step_size_texture = (cfg["snr"] * noise_texture_norm / grad_texture_norm) ** 2 * 2 * alpha
                grad_unfilled_texture = grad_texture * (1 - 1 * current_filled_mask)
                noise_unfilled_texture = noise_texture * (1 - 1 * current_filled_mask)
                current_unfilled_texture_mean = current_unfilled_texture + step_size_texture[:, None, None, None] * grad_texture
                current_unfilled_texture = current_unfilled_texture_mean + torch.sqrt(step_size_texture * 2)[:, None, None, None] * noise_texture
                current_large_texture = torch.where(current_filled_mask, current_filled_texture, current_unfilled_texture)
                current_face = renderer.render(texture=current_large_texture,
                                               background=background,
                                               pixel_uvs=pixel_uvs,
                                               background_mask=background_mask)
                render_func = functools.partial(renderer.render, background=background, pixel_uvs=pixel_uvs, background_mask=background_mask)

            if i % 10 == 0:
                save_images(_images=current_face, image_type="face", iteration=i, **save_images_kwargs)

        current_small_texture = get_current_small_texture(small_texture_size=cfg["small_texture_size"], large_texture_size=cfg["large_texture_size"], pixel_uvs=pixel_uvs, small_texture=current_small_texture, large_texture=current_large_texture)
        current_large_texture = get_current_large_texture(large_texture_size=cfg["large_texture_size"], small_texture=current_small_texture, initial_large_texture=initial_large_texture, initial_filled_mask=initial_filled_mask)
        final_face = renderer.render(texture=current_large_texture, background=target_background, pixel_uvs=pixel_uvs, background_mask=background_mask)
        save_images(_images=current_small_texture, image_type="final_small_texture", iteration=0, **save_images_kwargs)
        save_images(_images=current_large_texture, image_type="final_large_texture", iteration=0, **save_images_kwargs)
        save_images(_images=final_face, image_type="final_face", iteration=0, **save_images_kwargs)

    save_outputs(large_texture=current_large_texture, small_texture=current_small_texture, target_background=torch.zeros_like(target_background), renderer=renderer, elevs=elevs, azimuths=azimuths, image_save_dir=cfg["image_save_dir"], experiment_name=cfg["experiment_name"], obj_path=cfg["obj_path"], axis_angle_path=cfg["axis_angle_path"])
