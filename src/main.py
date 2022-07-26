from utils import set_experiment_name, save_cfg, set_seeds, get_initial_texture, update_texture, get_target_background, set_texture_and_obj_path
from model_utils import get_sde, get_score_model, get_reverse_diffusion_predictor_inpaint_update_fn, get_langevin_corrector_inpaint_update_fn
from visualization import save_images, save_optimization_gif, save_outputs
from renderer import Renderer

import itertools
import torch
from tqdm import tqdm

cfg = {
    "seed": 42,
    "batch_size": 1,
    "image_size": 256,
    "texture_size": 256,
    "saved_image_size": 256,
    "num_channels": 3,
    "image_save_dir": "../results",
    "device": torch.device("cuda:0"),
    "num_corrector_steps": 1,
    "sde_N": 1000,
    "snr": 0.075,
    "subject_id": 33673,
    "3dmm": "tf_flame"
}
set_experiment_name(cfg)
set_texture_and_obj_path(cfg)
save_cfg(cfg)
set_seeds(cfg)

sde = get_sde(cfg=cfg)
score_model = get_score_model(cfg=cfg)
predictor_inpaint_update_fn = get_reverse_diffusion_predictor_inpaint_update_fn(cfg=cfg, sde=sde)
corrector_inpaint_update_fn = get_langevin_corrector_inpaint_update_fn(cfg=cfg, sde=sde)

renderer = Renderer(cfg=cfg)

target_background = get_target_background(cfg)
current_texture = get_initial_texture(cfg)
timesteps = torch.linspace(sde.T, 1e-5, sde.N)

update_rounds = [0]
elevs = [0.0, -10.0, 10.0, -20.0, 20.0, -30.0, 30.0, -40.0, 40.0]
azimuths = [-50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
loop_tuples = list(itertools.product(update_rounds, elevs, azimuths))


with torch.no_grad():
    for update_round, elev, azimuth in tqdm(loop_tuples):
        save_images_kwargs = {"cfg": cfg, "elev": elev, "azimuth": azimuth, "update_round": update_round}

        pixel_uvs, background_mask, current_unoptimized_face, filled_mask, unfilled_mask = renderer.render(texture=current_texture, background=target_background, elev=elev, azimuth=azimuth, result_keys=["pixel_uvs", "background_mask", "face", "filled_mask", "unfilled_mask"])
        current_optimized_face = current_unoptimized_face * filled_mask + sde.prior_sampling(current_unoptimized_face.shape).to(current_unoptimized_face.device) * unfilled_mask

        save_images(_images=current_texture, image_type="initial_texture", iteration=0, **save_images_kwargs)
        save_images(_images=current_unoptimized_face, image_type="initial_face", iteration=0, **save_images_kwargs)
        save_images(_images=filled_mask, image_type="filled_mask", iteration=0, **save_images_kwargs)
        save_images(_images=unfilled_mask, image_type="unfilled_mask", iteration=0, **save_images_kwargs)

        for i in range(sde.N):
            t = timesteps[i]
            current_optimized_face, current_optimized_face_mean = predictor_inpaint_update_fn(score_model, current_unoptimized_face, filled_mask, current_optimized_face, t)
            current_optimized_face, current_optimized_face_mean = corrector_inpaint_update_fn(score_model, current_unoptimized_face, filled_mask, current_optimized_face, t)
            if i % 10 == 0:
                save_images(_images=current_optimized_face_mean, image_type="face", iteration=i, **save_images_kwargs)

        current_texture = update_texture(cfg=cfg, current_texture=current_texture, pixel_uvs=pixel_uvs.clone(), current_optimized_face_mean=current_optimized_face_mean.clone(), unfilled_mask=unfilled_mask, update_round=update_round)
        current_optimized_face = renderer.render_prerendering_results(texture=current_texture, background=torch.zeros_like(target_background), pixel_uvs=pixel_uvs, background_mask=background_mask)

        save_images(_images=current_texture, image_type="final_texture", iteration=0, **save_images_kwargs)
        save_images(_images=current_optimized_face, image_type="final_face", iteration=0, **save_images_kwargs)

        save_optimization_gif(**save_images_kwargs)

    save_outputs(cfg=cfg, final_texture=current_texture, target_background=torch.zeros_like(target_background), renderer=renderer, elevs=elevs, azimuths=azimuths)
