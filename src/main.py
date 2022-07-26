from utils import set_experiment_name, save_cfg, set_seeds, get_initial_texture, get_grad_texture, get_target_background, set_texture_and_obj_path
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
    "texture_size": 2048,
    "saved_image_size": 256,
    "num_channels": 3,
    "image_save_dir": "../results",
    "device": torch.device("cuda:0"),
    "num_corrector_steps": 6,
    "sde_N": 1000,
    "snr": 0.015, # changed from 0.075
    "subject_id": 33673,
    "3dmm": "tf_flame"
}
set_experiment_name(cfg)
set_texture_and_obj_path(cfg)
save_cfg(cfg)
set_seeds(cfg)

sde = get_sde(cfg=cfg)
score_model = get_score_model(cfg=cfg)
score_fn = get_score_fn(score_model=score_model, sde=sde)

renderer = Renderer(cfg=cfg)

target_background = get_target_background(cfg)
initial_texture = get_initial_texture(cfg)
filled_mask = (initial_texture.sum(axis=1) != 3).unsqueeze(1).repeat(repeats=[1, 3, 1, 1])
current_unfilled_texture = torch.randn_like(initial_texture) * sde.sigma_max * (1 - 1 * filled_mask)
pixel_uvs, background_mask = renderer.prerender(texture=initial_texture, elev=0.0, azimuth=0.0, result_keys=["pixel_uvs", "background_mask"])
initial_face = renderer.render(texture=initial_texture, background=target_background, pixel_uvs=pixel_uvs, background_mask=background_mask)
save_images(_images=initial_face, image_type="initial_face", iteration=0, cfg=cfg, elev=0.0, azimuth=0.0)
save_images(_images=initial_texture, image_type="initial_texture", iteration=0, cfg=cfg, elev=0.0, azimuth=0.0)
save_images(_images=current_unfilled_texture, image_type="initial_unfilled_texture", iteration=0, cfg=cfg, elev=0.0, azimuth=0.0)
save_images(_images=filled_mask, image_type="filled_mask", iteration=0, cfg=cfg, elev=0.0, azimuth=0.0)

timesteps = torch.linspace(sde.T, 1e-5, sde.N)

elevs = [0.0] # [0.0, -10.0, 10.0, -20.0, 20.0, -30.0, 30.0, -40.0, 40.0]
azimuths = [0.0] # [-50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]

prerender_results = {}
loop_tuples = list(itertools.product(elevs, azimuths))
for elev, azimuth in loop_tuples:
    pixel_uvs, background_mask = renderer.prerender(texture=initial_texture, elev=elev, azimuth=azimuth, result_keys=["pixel_uvs", "background_mask"])
    prerender_results[f"{elev}_{azimuth}"] = {
        "pixel_uvs": pixel_uvs,
        "background_mask": background_mask
    }


with torch.no_grad():
    for i in tqdm(list(range(sde.N))):
        t = timesteps[i]

        vec_t = torch.ones(cfg["batch_size"], device=cfg["device"]) * t
        alpha = torch.ones_like(vec_t)

        background_mean, background_std = sde.marginal_prob(target_background, vec_t)
        background = background_mean + torch.randn_like(background_mean) * background_std[:, None, None, None]

        current_filled_texture_mean, current_filled_texture_std = sde.marginal_prob(initial_texture, vec_t)
        current_filled_texture = current_filled_texture_mean + torch.randn_like(current_filled_texture_mean) * current_filled_texture_std[:, None, None, None]

        for elev, azimuth in loop_tuples:
            save_images_kwargs = {"cfg": cfg, "elev": elev, "azimuth": azimuth}
            current_prerender_results = prerender_results[f"{elev}_{azimuth}"]
            pixel_uvs, background_mask = current_prerender_results["pixel_uvs"], current_prerender_results["background_mask"]
            for j in range(cfg["num_corrector_steps"]):
                current_texture = torch.where(filled_mask, current_filled_texture, current_unfilled_texture)
                current_face = renderer.render(texture=current_texture,
                                               background=background,
                                               pixel_uvs=pixel_uvs,
                                               background_mask=background_mask)
                render_func = functools.partial(renderer.render, background=background, pixel_uvs=pixel_uvs, background_mask=background_mask)
                grad_face = score_fn(current_face, vec_t)
                grad_texture = get_grad_texture(texture=current_texture, grad_face=grad_face, render_func=render_func)
                grad_texture_norm = torch.norm(grad_texture.reshape(grad_texture.shape[0], -1), dim=-1).mean()
                noise_texture = torch.randn_like(grad_texture)
                noise_texture_norm = torch.norm(noise_texture.reshape(noise_texture.shape[0], -1), dim=-1).mean()
                step_size_texture = (cfg["snr"] * noise_texture_norm / grad_texture_norm) ** 2 * 2 * alpha
                grad_unfilled_texture = grad_texture * (1 - 1 * filled_mask)
                noise_unfilled_texture = noise_texture * (1 - 1 * filled_mask)
                current_unfilled_texture_mean = current_unfilled_texture + step_size_texture[:, None, None, None] * grad_texture
                current_unfilled_texture = current_unfilled_texture_mean + torch.sqrt(step_size_texture * 2)[:, None, None, None] * noise_texture

            if i % 10 == 0:
                save_images(_images=current_face, image_type="face", iteration=i, **save_images_kwargs)
                save_images(_images=current_texture, image_type="texture", iteration=i, **save_images_kwargs)

    save_outputs(cfg=cfg, final_texture=current_texture, target_background=torch.zeros_like(target_background), renderer=renderer, elevs=elevs, azimuths=azimuths)
