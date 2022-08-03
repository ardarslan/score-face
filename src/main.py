from utils import set_experiment_name, set_seeds, set_3dmm_result_paths, \
                  save_cfg, get_target_background, get_initial_textures, \
                  get_filled_mask, get_cfg, get_dark_pixel_allowances \
                  # set_device
from model_utils import get_sde, get_score_model, get_score_fn, get_langevin_corrector_inpaint_update_fn, get_reverse_diffusion_predictor_inpaint_update_fn
from visualization import save_outputs
from optimization import run_single_view_optimization_in_texture_space, run_single_view_optimization_in_image_space
from rendering import Renderer
from transformation import get_initial_quaternion_and_inverse, get_elev_azimuth_quaternion_tuples

import itertools

import torch
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    cfg = get_cfg()
    set_seeds(cfg=cfg)
    # set_device(cfg=cfg)
    set_experiment_name(cfg=cfg)
    set_3dmm_result_paths(cfg=cfg)
    save_cfg(cfg=cfg)

    sde = get_sde(cfg=cfg)
    score_model = get_score_model(cfg=cfg)
    score_fn = get_score_fn(score_model=score_model, sde=sde)

    renderer = Renderer(cfg=cfg)
    initial_quaternion, inverse_of_initial_quaternion = get_initial_quaternion_and_inverse(cfg=cfg)

    timesteps = torch.linspace(sde.T, 1e-5, sde.N)
    elevs = list(np.arange(start=cfg["min_elev"], stop=cfg["max_elev"]+cfg["step_elev"], step=cfg["step_elev"]))
    azimuths = list(np.arange(start=cfg["min_azimuth"], stop=cfg["max_azimuth"]+cfg["step_azimuth"], step=cfg["step_azimuth"]))
    ordered_elev_azimuth_quaternion_tuples = get_elev_azimuth_quaternion_tuples(cfg=cfg, elevs=elevs, azimuths=azimuths, initial_quaternion=initial_quaternion, inverse_of_initial_quaternion=inverse_of_initial_quaternion)
    dark_pixel_allowances = get_dark_pixel_allowances(cfg=cfg)

    target_background = get_target_background(cfg=cfg)
    initial_small_texture, initial_large_texture = get_initial_textures(cfg=cfg)
    current_small_texture = initial_small_texture.clone()

    optimization_space = cfg["optimization_space"]
    if optimization_space == "texture":
        initial_filled_texture_mask = get_filled_mask(cfg=cfg, image=initial_large_texture)
        current_large_texture = initial_large_texture.clone()
    elif optimization_space == "image":
        predictor_inpaint_update_fn = get_reverse_diffusion_predictor_inpaint_update_fn(sde=sde)
        corrector_inpaint_update_fn = get_langevin_corrector_inpaint_update_fn(cfg=cfg, sde=sde)

    with torch.no_grad():
        for dark_pixel_allowance, (elev, azimuth, quaternion) in tqdm(list(itertools.product(dark_pixel_allowances, ordered_elev_azimuth_quaternion_tuples))):
            if optimization_space == "texture":
                current_small_texture, current_large_texture = run_single_view_optimization_in_texture_space(cfg=cfg, current_small_texture=current_small_texture, current_large_texture=current_large_texture, initial_large_texture=initial_large_texture, initial_filled_texture_mask=initial_filled_texture_mask, sde=sde, timesteps=timesteps, elev=elev, azimuth=azimuth, quaternion=quaternion, dark_pixel_allowance=dark_pixel_allowance, renderer=renderer, target_background=target_background, score_fn=score_fn)
            elif optimization_space == "image":
                current_small_texture = run_single_view_optimization_in_image_space(cfg=cfg, elev=elev, azimuth=azimuth, quaternion=quaternion, dark_pixel_allowance=dark_pixel_allowance, renderer=renderer, target_background=target_background, sde=sde, current_small_texture=current_small_texture, timesteps=timesteps, score_model=score_model, predictor_inpaint_update_fn=predictor_inpaint_update_fn, corrector_inpaint_update_fn=corrector_inpaint_update_fn)

    if optimization_space == "texture":
        final_texture = current_large_texture
    elif optimization_space == "image":
        final_texture = current_small_texture

    save_outputs(cfg=cfg, final_texture=final_texture, target_background=torch.zeros_like(target_background), renderer=renderer, elevs=elevs, azimuths=azimuths, inverse_of_initial_quaternion=inverse_of_initial_quaternion)
