from utils import set_experiment_name, set_seeds, set_3dmm_result_paths, \
                  save_cfg, get_target_background, get_cfg, get_dark_pixel_alloweds, \
                  set_device, set_image_size_and_checkpoint_path
from model_utils import get_sde, get_score_model, get_score_fn
from optimization import run_multi_view_optimization_in_texture_space, run_multi_view_optimization_in_image_space
from rendering import Renderer
from transformation import get_ordered_prerender_results

import torch
import numpy as np


if __name__ == "__main__":
    cfg = get_cfg()
    set_seeds(cfg=cfg)
    set_device(cfg=cfg)
    set_experiment_name(cfg=cfg)
    set_3dmm_result_paths(cfg=cfg)
    set_image_size_and_checkpoint_path(cfg=cfg)
    save_cfg(cfg=cfg)

    sde = get_sde(cfg=cfg)
    score_model = get_score_model(cfg=cfg)
    score_fn = get_score_fn(score_model=score_model, sde=sde)

    renderer = Renderer(cfg=cfg)

    timesteps = torch.linspace(sde.T, 1e-5, sde.N)
    elevs = list(np.arange(start=cfg["min_elev"], stop=cfg["max_elev"]+cfg["step_elev"], step=cfg["step_elev"]))
    azimuths = list(np.arange(start=cfg["min_azimuth"], stop=cfg["max_azimuth"]+cfg["step_azimuth"], step=cfg["step_azimuth"]))
    ordered_prerender_results = get_ordered_prerender_results(cfg=cfg, elevs=elevs, azimuths=azimuths, renderer=renderer)

    dark_pixel_alloweds = get_dark_pixel_alloweds(cfg=cfg)
    target_background = get_target_background(cfg=cfg)

    if cfg["optimization_space"] == "texture":
        run_multi_view_optimization_in_texture_space(cfg=cfg, renderer=renderer, target_background=target_background, sde=sde, timesteps=timesteps, score_fn=score_fn, dark_pixel_alloweds=dark_pixel_alloweds, elevs=elevs, azimuths=azimuths, ordered_prerender_results=ordered_prerender_results)
    elif cfg["optimization_space"] == "image":
        run_multi_view_optimization_in_image_space(cfg=cfg, renderer=renderer, target_background=target_background, sde=sde, timesteps=timesteps, score_model=score_model, dark_pixel_alloweds=dark_pixel_alloweds, elevs=elevs, azimuths=azimuths, ordered_prerender_results=ordered_prerender_results)
    else:
        raise Exception(f"Not a valid 'optimization_space': {cfg['optimization_space']}.")
