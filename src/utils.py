import os
import cv2
import random
import time
import pprint
import argparse
from uuid import uuid4
from typing import Dict, Any, Tuple, List

import torch
import numpy as np


def get_cfg() -> Dict[str, Any]:
    """Initialize CLI argument parser."""
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument("--input_obj_path", type=str, required=True, help="Absolute path to input .obj file. There should be a .png file and _axis_angle.npy file in the same directory with the provided .obj file.")
    parser.add_argument("--optimization_space", type=str, required=True, choices=["image", "texture"], help="Which space to use for optimization. It should be 'image' or 'texture'.")
    parser.add_argument("--num_corrector_steps", type=int, required=True, help="Number of correct steps at each noise level. Use 1 for image space optimization, and use 6 for texture space optimization.")
    parser.add_argument("--snr", type=float, required=True, help="SNR value which acts like a step size during the optimization process. Use 0.075 for image space optimization, and 0.015 for texture space optimization.")
    parser.add_argument("--order_views", type=str, required=True, choices=["true", "false", "True", "False", "TRUE", "FALSE"], help="When this is 'true', the views are ordered so that the optimization starts with the closest global rotation to the one in the original image, and it ends with the furthest one.")
    parser.add_argument("--two_rounds", type=str, required=True, choices=["true", "false", "True", "False", "TRUE", "FALSE"], help="When this is 'true', the pipeline runs two optimization rounds. In the first round, if a pixel is too dark (gray_intensity < min_gray_intensity_in_texture), then it is not copied to the small texture. In the second round, it is copied even though the pixel is dark. When this is 'false', the pipeline runs only the second round.")

    # arguments with default values
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--large_texture_size", type=int, default=2048, choices=[2048], help="Width and height of large texture images. If the optimizatioon is done in image space, this is not used. Only 2048 is supported.")
    parser.add_argument("--small_texture_size", type=int, default=512, choices=[512], help="Width and height of small texture images. If the optimization is done in image space, this is used as the texture size. Only 512 is supported.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use.")
    parser.add_argument("--batch_size", type=int, default=1, choices=[1], help="Currently only batch_size=1 is supported.")
    parser.add_argument("--num_channels", type=int, default=3, choices=[3], help="Currently only num_channels=3 is supported.")
    parser.add_argument("--results_dir", type=str, default="../results", help="Path to results directory.")
    parser.add_argument("--checkpoint_dir", type=str, default="../assets", help="Directory to score model's checkpoint files.")
    parser.add_argument("--sde_N", type=int, default=1000, help="Number of different noise levels used during the optimizations process.")
    parser.add_argument("--camera_distance", type=float, default=1.2, help="Distance between the rendered mesh and the camera.")
    parser.add_argument("--min_gray_intensity_in_texture", type=float, default=0.30, help="This is used only if 'two_rounds' argument is 'true'. This argument corresponds to the maximum gray intensity allowed for a pixel to be copied to the small texture during the first optimization round.")
    parser.add_argument("--initial_view", type=str, default="image", choices=["frontal", "image"], help="Initial view to optimize.")
    parser.add_argument("--min_elev", type=float, default=-10.0, help="Minimum elev used during the optimization process.")
    parser.add_argument("--max_elev", type=float, default=10.0, help="Maximum elev used during the optimization process.")
    parser.add_argument("--step_elev", type=float, default=10.0, help="Elev step used during the optimization process.")
    parser.add_argument("--min_azimuth", type=float, default=-40.0, help="Minimum azimuth used during the optimization process.")
    parser.add_argument("--max_azimuth", type=float, default=40.0, help="Maximum azimuth used during the optimization process.")
    parser.add_argument("--step_azimuth", type=float, default=20.0, help="Azimuth step used during the optimization process.")

    cfg = parser.parse_args().__dict__
    return cfg


def set_seeds(cfg: Dict[str, Any]) -> None:
    seed = cfg["seed"]
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_experiment_dir(cfg: Dict[str, Any]) -> str:
    results_dir = cfg["results_dir"]
    experiment_name = cfg["experiment_name"]
    experiment_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def set_device(cfg: Dict[str, Any]) -> None:
    if cfg["device"] == "cuda" and torch.cuda.is_available():
        cfg["device"] = torch.device("cuda")
    else:
        cfg["device"] = torch.device("cpu")
    print(f"Using device {cfg['device']}.")


def save_cfg(cfg: Dict[str, Any]) -> None:
    results_dir = get_experiment_dir(cfg=cfg)
    cfg_path = os.path.join(results_dir, "cfg.txt")
    with open(cfg_path, "w") as cfg_writer:
        cfg_writer.write(pprint.pformat(cfg, indent=4))


def set_experiment_name(cfg: Dict[str, Any]) -> None:
    cfg["experiment_name"] = f"{int(time.time())}_{uuid4().hex}"


def get_initial_textures(cfg: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    input_texture_path = cfg["input_texture_path"]
    small_texture_size = cfg["small_texture_size"]
    large_texture_size = cfg["large_texture_size"]
    device = cfg["device"]

    texture = cv2.imread(input_texture_path)
    gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
    # if _3dmm == "deca":
    #     black_mask = gray <= 0
    #     black_mask_moved = np.vstack((np.zeros((1, black_mask.shape[1])), black_mask[:-1, :]))
    #     black_mask = np.logical_or(black_mask, black_mask_moved)
    # elif _3dmm == "tf_flame":
    black_mask = (gray <= 50)
    # else:
    #     raise Exception(f"Not a valid _3dmm {_3dmm}.")
    black_mask = np.tile(black_mask[:, :, None], reps=[1, 1, 3])
    texture = np.clip(texture / 255 + black_mask, 0, 1)
    texture = texture[:, :, [2, 1, 0]]

    if texture.shape[0] != small_texture_size:
        initial_small_texture = cv2.resize(texture, dsize=(small_texture_size, small_texture_size), interpolation=cv2.INTER_NEAREST)
    else:
        initial_small_texture = texture.copy()
    initial_small_texture = torch.tensor(initial_small_texture, device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    if texture.shape[0] != large_texture_size:
        initial_large_texture = cv2.resize(texture, dsize=(large_texture_size, large_texture_size), interpolation=cv2.INTER_NEAREST)
    else:
        initial_large_texture = texture.copy()
    initial_large_texture = torch.tensor(initial_large_texture, device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return initial_small_texture, initial_large_texture


def set_3dmm_result_paths(cfg: Dict[str, Any]) -> None:
    # if cfg["3dmm"] == "deca":
    #     cfg["texture_path"] = f"/local/home/aarslan/DECA/TestSamples/examples/results/{cfg['subject_id']}/{cfg['subject_id']}.png"
    #     cfg["obj_path"] = f"/local/home/aarslan/DECA/TestSamples/examples/results/{cfg['subject_id']}/{cfg['subject_id']}.obj"
    #     raise Exception("We need to output rotation vector from DECA.")
    # elif cfg["3dmm"] == "tf_flame":
    cfg["input_texture_path"] = cfg["input_obj_path"].replace(".obj", ".png")
    cfg["input_axis_angle_path"] = cfg["input_obj_path"].replace(".obj", "_axis_angle.npy")


def set_image_size_and_checkpoint_path(cfg: Dict[str, Any]) -> None:
    if cfg["optimization_space"] == "image":
        cfg["image_size"] = 1024
        cfg["checkpoint_path"] = os.path.join(cfg["checkpoint_dir"], "checkpoint_60.pth")
    elif cfg["optimization_space"] == "texture":
        cfg["image_size"] = 256
        cfg["checkpoint_path"] = os.path.join(cfg["checkpoint_dir"], "checkpoint_48.pth")
    else:
        raise Exception(f"Not a valid optimization_space {cfg['optimization_space']}.")


def get_target_background(cfg: Dict[str, Any]) -> torch.Tensor:
    batch_size = cfg["batch_size"]
    num_channels = cfg["num_channels"]
    image_size = cfg["image_size"]
    device = cfg["device"]

    image_shape = (batch_size, num_channels, image_size, image_size)
    target_background = torch.zeros(size=image_shape, device=device)
    return target_background


def get_filled_mask(cfg: Dict[str, Any], image: torch.Tensor) -> torch.Tensor:
    num_channels = cfg["num_channels"]

    return (image.sum(axis=1) != num_channels).unsqueeze(1).repeat(repeats=[1, num_channels, 1, 1])


def get_dark_pixel_alloweds(cfg: Dict[str, Any]) -> List[bool]:
    two_rounds = cfg["two_rounds"]

    if two_rounds.lower() == "true":
        dark_pixel_alloweds = [False, True]
    elif two_rounds.lower() == "false":
        dark_pixel_alloweds = [True]
    return dark_pixel_alloweds
