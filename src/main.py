import torch
import itertools
import functools
from tqdm import tqdm

from utils import (
    get_argument_parser,
    set_seeds,
    get_experiment_name,
    save_cfg
)
from model_utils import (
    get_score_model,
    get_sde,
    get_score_fn,
    get_grad_texture_and_background
)
from renderer import Renderer
from visualization import save_images
from random import shuffle


def main(cfg):
    if cfg["predict"] == "texture_and_background" and cfg["render_mode"] == "3d":
        raise Exception(f"Cannot predict 'texture_and_background' when render_mode is '3d'.")
    
    if cfg["add_noise"] == "face" and cfg["render_mode"] == "3d_fixed_background":
        raise Exception(f"Cannot add noise to 'face' when render_mode is '3d_fixed_background'.")

    cfg["experiment_name"] = get_experiment_name()
    if torch.cuda.is_available():
        cfg["device"] = torch.device(f"cuda:{cfg['cuda_device']}")
        torch.cuda.set_device(cfg["device"])
    else:
        cfg["device"] = torch.device("cpu")
    print(f"Using {cfg['device']}.")

    save_cfg(cfg)
    set_seeds(cfg)

    score_model = get_score_model(cfg)
    sde = get_sde()
    score_fn = get_score_fn(score_model=score_model, sde=sde)
    renderer = Renderer(cfg)

    with torch.no_grad():
        image_shape = (cfg["batch_size"], cfg["num_channels"], cfg["image_size"], cfg["image_size"])
        texture_shape = (cfg["batch_size"], cfg["num_channels"], cfg["texture_size"], cfg["texture_size"])

        texture = torch.randn(size=texture_shape, device=cfg["device"]) * cfg["texture_initial_noise_coefficient"]
        if cfg["render_mode"] == "3d_fixed_background":
            background = torch.zeros(size=image_shape, device=cfg["device"])
        else:
            background = torch.randn(size=image_shape, device=cfg["device"]) * cfg["background_initial_noise_coefficient"]
        elev, azimuth = renderer.get_random_elev_azimuth()
        face = renderer.render(texture=texture, background=background, elev=elev, azimuth=azimuth, return_face_mean=True, j=None, rsde=None, vec_t=None, z=None, noise_face=None, step_size_face=None)
        render_func = functools.partial(renderer.render, elev=elev, azimuth=azimuth, return_face_mean=True, j=None, rsde=None, vec_t=None, z=None, noise_face=None, step_size_face=None)

        timesteps = torch.linspace(cfg["sde_T"], cfg["sampling_eps"], cfg["sde_N"], device=cfg["device"])
        rsde = sde.reverse(score_fn, cfg["probability_flow"])
        alpha = torch.ones(cfg["batch_size"], device=cfg["device"])

        for i in tqdm(list(range(cfg["sde_N"]))):
            t = timesteps[i]
            vec_t = torch.ones(cfg["batch_size"], device=cfg["device"]) * t
            labels = sde.marginal_prob(torch.zeros(size=image_shape), vec_t)[1]

            for j in range(cfg["num_corrector_steps"]):
                grad_face = score_model(face, labels)
                grad_face_norm = torch.norm(grad_face.reshape(grad_face.shape[0], -1), dim=-1).mean()
                noise_face = torch.randn(size=image_shape, device=cfg["device"])
                noise_face_norm = torch.norm(noise_face.reshape(noise_face.shape[0], -1), dim=-1).mean()
                step_size_face = (cfg["background_snr"] * noise_face_norm / grad_face_norm) ** 2 * 2 * alpha

                grad_texture, grad_background = get_grad_texture_and_background(texture=texture, background=background, grad_face=grad_face, render_func=render_func)

                grad_texture_norm = torch.norm(grad_texture.reshape(grad_texture.shape[0], -1), dim=-1).mean()
                noise_texture = torch.randn(size=texture_shape, device=cfg["device"])
                noise_texture_norm = torch.norm(noise_texture.reshape(noise_texture.shape[0], -1), dim=-1).mean()
                step_size_texture = (cfg["texture_snr"] * noise_texture_norm / grad_texture_norm) ** 2 * 2 * alpha
                texture_mean = texture + step_size_texture[:, None, None, None] * grad_texture

                if cfg["render_mode"] != "3d_fixed_background":
                    grad_background_norm = torch.norm(grad_background.reshape(grad_background.shape[0], -1), dim=-1).mean()
                    noise_background = torch.randn(size=image_shape, device=cfg["device"])
                    noise_background_norm = torch.norm(noise_background.reshape(noise_background.shape[0], -1), dim=-1).mean()
                    step_size_background = (cfg["background_snr"] * noise_background_norm / grad_background_norm) ** 2 * 2 * alpha
                    background_mean = background + step_size_background[:, None, None, None] * grad_background

                    if cfg["add_noise"] in ["face", "never"]:
                        background = background_mean
                    elif cfg["add_noise"] == "texture_and_background":
                        background = background_mean + torch.sqrt(step_size_background * 2)[:, None, None, None] * noise_background * cfg["background_noise_coefficient"]
                    else:
                        raise Exception(f"Not a valid add_noise {cfg['add_noise']}.")
                
                if cfg["add_noise"] in ["face", "never"]:
                    texture = texture_mean
                elif cfg["add_noise"] == "texture_and_background":
                    texture = texture_mean + torch.sqrt(step_size_texture * 2)[:, None, None, None] * noise_texture * cfg["texture_noise_coefficient"]
                else:
                    raise Exception(f"Not a valid add_noise {cfg['add_noise']}.")

                elev, azimuth = renderer.get_random_elev_azimuth()
                z = torch.randn(size=image_shape, device=cfg["device"])

                if cfg["predict"] == "texture_and_background":
                    f, G = rsde.discretize(face, vec_t)
                    texture = texture - f
                    texture = texture + G[:, None, None, None] * z

                    if cfg["render_mode"] != "3d_fixed_background":
                        background = background - f
                        background = background + G[:, None, None, None] * z

                face = renderer.render(texture=texture, background=background, elev=elev, azimuth=azimuth, return_face_mean=False, j=j, rsde=rsde, vec_t=vec_t, z=z, noise_face=noise_face, step_size_face=step_size_face)
                render_func = functools.partial(renderer.render, elev=elev, azimuth=azimuth, return_face_mean=False, j=j, rsde=rsde, vec_t=vec_t, z=z, noise_face=noise_face, step_size_face=step_size_face)

            if (i + 1) % 10 == 0:
                face = renderer.render(texture=texture, background=background, elev=elev, azimuth=azimuth, return_face_mean=True, j=None, rsde=None, vec_t=None, z=None, noise_face=None, step_size_face=None)
                save_images(cfg, face, "face", i)
                save_images(cfg, texture, "texture", i)
                save_images(cfg, background, "background", i)


if __name__ == "__main__":
    cfg = get_argument_parser().parse_args().__dict__

    cfg["add_noise"] = "texture_and_background"

    cfg["background_noise_coefficient"] = 1
    cfg["background_snr"] = 0.2
    cfg["background_initial_noise_coefficient"] = 10

    texture_noise_coefficients = [0.01, 0.05, 0.1, 0.5, 1.0]
    texture_snrs = [0.01, 0.025, 0.05, 0.075, 0.1]
    texture_initial_noise_coefficients = [0.01, 0.1, 1.0, 10, 100]

    hyperparameters = list(itertools.product(texture_noise_coefficients, texture_snrs, texture_initial_noise_coefficients))
    shuffle(hyperparameters)

    for texture_noise_coefficient in texture_noise_coefficients:
        for texture_snr in texture_snrs:
            for texture_initial_noise_coefficient in texture_initial_noise_coefficients:
                cfg["texture_noise_coefficient"] = texture_noise_coefficient
                cfg["texture_snr"] = texture_snr
                cfg["texture_initial_noise_coefficient"] = texture_initial_noise_coefficient
                main(cfg)
