import os
import cv2
import math
import random
import time
import pprint
import argparse
import itertools
from uuid import uuid4

import torch
import numpy as np
from pytorch3d.transforms import axis_angle_to_quaternion, matrix_to_quaternion, quaternion_multiply, quaternion_invert


def get_cfg() -> dict:
    """Initialize CLI argument parser."""
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument("--input_obj_path", type=str, required=True, help="Absolute path to input .obj file. There should be a .png file and _axis_angle.npy file in the same directory with the provided .obj file.")
    parser.add_argument("--optimization_space", type=str, required=True, choices=["image", "texture"], help="Which space to use for optimization. It should be 'image' or 'texture'.")
    parser.add_argument("--num_corrector_steps", type=int, required=True, help="Number of correct steps at each noise level. Use 1 for image space optimization, and use 6 for texture space optimization.")
    parser.add_argument("--snr", type=float, required=True, help="SNR value which acts like a step size during the optimization process. Use 0.075 for image space optimization, and 0.015 for image space optimization.")

    # arguments with default values
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--image_size", type=int, default=256, help="Width and height of face images.")
    parser.add_argument("--large_texture_size", type=int, default=2048, choices=[1536, 2048], help="Width and height of large texture images. If the optimizatioon is done in image space, this is not used.")
    parser.add_argument("--small_texture_size", type=int, default=256, choices=[256, 512], help="Width and height of small texture images. If the optimization is done in image space, this is used as the texture size.")
    parser.add_argument("--saved_image_size", type=int, default=256, help="Width and height of face images saved during the optimization.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use.")
    parser.add_argument("--batch_size", type=int, default=1, choices=[1], help="Currently only batch_size=1 is supported.")
    parser.add_argument("--num_channels", type=int, default=3, choices=[3], help="Currently only num_channels=3 is supported.")
    parser.add_argument("--results_dir", type=str, default="../results", help="Path to results directory.")
    parser.add_argument("--checkpoint_path", type=str, default="../assets/checkpoint_48.pth", help="Path to score model's checkpoint file.")
    parser.add_argument("--sde_N", type=int, default=1000, help="Number of different noise levels used during the optimizations process.")
    parser.add_argument("--camera_distance", type=float, default=1.2, help="Distance between the rendered mesh and the camera.")
    parser.add_argument("--optimization_rounds", type=list, default=[0, 1], help="List of optimization round indices. In round 0, if a pixel is too dark (gray_intensity < max_gray_intensity_in_round_0), then it is not copied to the small texture. In round 1, it is copied even though the pixel is dark.")
    parser.add_argument("--max_gray_intensity_in_round_0", type=float, default=0.25, help="Maximum gray intensity allowed for a pixel to be copied to the small texture during optimization round 0.")
    parser.add_argument("--min_elev", type=int, default=-20, help="Minimum elev used during the optimization process.")
    parser.add_argument("--max_elev", type=int, default=20, help="Maximum elev used during the optimization process.")
    parser.add_argument("--step_elev", type=int, default=5, help="Elev step used during the optimization process.")
    parser.add_argument("--min_azimuth", type=int, default=-40, help="Minimum azimuth used during the optimization process.")
    parser.add_argument("--max_azimuth", type=int, default=40, help="Maximum azimuth used during the optimization process.")
    parser.add_argument("--step_azimuth", type=int, default=5, help="Azimuth step used during the optimization process.")

    cfg = parser.parse_args().__dict__
    return cfg


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_experiment_dir(results_dir, experiment_name):
    experiment_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def set_device(cfg):
    if cfg["device"] == "cuda" and torch.cuda.is_available():
        cfg["device"] = torch.device("cuda")
    else:
        cfg["device"] = torch.device("cpu")


def save_cfg(cfg):
    results_dir = get_experiment_dir(results_dir=cfg["results_dir"], experiment_name=cfg["experiment_name"])
    cfg_path = os.path.join(results_dir, "cfg.txt")
    with open(cfg_path, "w") as cfg_writer:
        cfg_writer.write(pprint.pformat(cfg, indent=4))


def set_experiment_name(cfg):
    cfg["experiment_name"] = f"{int(time.time())}_{uuid4().hex}"


def get_initial_textures(input_texture_path, small_texture_size, large_texture_size, device):
    texture = cv2.imread(input_texture_path)
    gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
    # if _3dmm == "deca":
    #     black_mask = gray <= 0
    #     black_mask_moved = np.vstack((np.zeros((1, black_mask.shape[1])), black_mask[:-1, :]))
    #     black_mask = np.logical_or(black_mask, black_mask_moved)
    # elif _3dmm == "tf_flame":
    black_mask = (gray <= 40)
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


def get_current_small_texture(small_texture_size, large_texture_size, pixel_uvs, small_texture, large_texture, optimization_round, max_gray_intensity_in_round_0):
    large_texture_clamped = torch.clamp(large_texture, min=0.0, max=1.0)
    for sample_idx in range(pixel_uvs.shape[0]):
        gray = cv2.cvtColor(large_texture_clamped[sample_idx, :, :, :].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2GRAY)
        for y in range(pixel_uvs.shape[1]):
            for x in range(pixel_uvs.shape[2]):
                u_small = math.floor((pixel_uvs[sample_idx, y, x, 0] + 1) * small_texture_size / 2)
                v_small = small_texture_size - 1 - math.floor(small_texture_size * (pixel_uvs[sample_idx, y, x, 1] + 1) / 2)
                u_large = math.floor((pixel_uvs[sample_idx, y, x, 0] + 1) * large_texture_size / 2)
                v_large = large_texture_size - 1 - math.floor(large_texture_size * (pixel_uvs[sample_idx, y, x, 1] + 1) / 2)
                if ((u_small == 0) and (v_small == small_texture_size - 1)):
                    continue
                elif optimization_round == 0 and gray[v_large, u_large] < max_gray_intensity_in_round_0:
                    continue
                else:
                    small_texture[sample_idx, :, v_small, u_small] = large_texture_clamped[sample_idx, :, v_large, u_large]
    return small_texture


def get_current_large_texture(large_texture_size, small_texture, initial_large_texture, initial_filled_texture_mask):
    upsampled_small_texture = torch.nn.functional.interpolate(input=small_texture, size=large_texture_size, mode='nearest')
    return torch.where(initial_filled_texture_mask, initial_large_texture, upsampled_small_texture)


def set_3dmm_result_paths(cfg):
    # if cfg["3dmm"] == "deca":
    #     cfg["texture_path"] = f"/local/home/aarslan/DECA/TestSamples/examples/results/{cfg['subject_id']}/{cfg['subject_id']}.png"
    #     cfg["obj_path"] = f"/local/home/aarslan/DECA/TestSamples/examples/results/{cfg['subject_id']}/{cfg['subject_id']}.obj"
    #     raise Exception("We need to output rotation vector from DECA.")
    # elif cfg["3dmm"] == "tf_flame":
    cfg["input_texture_path"] = cfg["input_obj_path"].replace(".obj", ".png")
    cfg["input_axis_angle_path"] = cfg["input_obj_path"].replace(".obj", "_axis_angle.npy")


def get_target_background(batch_size, num_channels, image_size, device):
    image_shape = (batch_size, num_channels, image_size, image_size)
    target_background = torch.zeros(size=image_shape, device=device)
    return target_background


def get_grad_texture(texture, grad_face, render_func):
    _, grad_texture = torch.autograd.functional.vjp(func=render_func, inputs=texture, v=grad_face, create_graph=False, strict=False)
    return grad_texture


def get_filled_mask(image, num_channels):
    return (image.sum(axis=1) != num_channels).unsqueeze(1).repeat(repeats=[1, num_channels, 1, 1])


def get_initial_quaternion_and_inverse(input_axis_angle_path, device):
    axis_angle = torch.tensor(np.load(input_axis_angle_path), device=device)
    quaternion = axis_angle_to_quaternion(axis_angle)
    quaternion_fix = matrix_to_quaternion(axis_angle_to_matrix(axis="Y", angle=torch.tensor([[np.pi]], device=device)))
    inverse_of_initial_quaternion = quaternion_multiply(quaternion, quaternion_fix)
    initial_quaternion = quaternion_invert(quaternion)
    return initial_quaternion, inverse_of_initial_quaternion


def get_angle_between_two_quaternions(quaternion_1, quaternion_2):
    """
    Taken from https://math.stackexchange.com/a/90098
    """

    dot_product = (quaternion_1 * quaternion_2).sum()
    squared_dot_product = dot_product * dot_product
    cos = 2 * squared_dot_product - 1
    angle = float(torch.arccos(cos))
    return angle


def get_ordered_elev_azimuth_quaternion_tuples(elevs, azimuths, initial_quaternion, inverse_of_initial_quaternion, device):
    elevs_azimuths = list(itertools.product(elevs, azimuths))
    results = []
    for elev, azimuth in elevs_azimuths:
        quaternion_elev = matrix_to_quaternion(axis_angle_to_matrix(axis="X", angle=torch.tensor([[np.pi * elev / 180.0]], device=device)))
        quaternion_azimuth = matrix_to_quaternion(axis_angle_to_matrix(axis="Y", angle=torch.tensor([[np.pi * azimuth / 180.0]], device=device)))
        quaternion_elev_azimuth = quaternion_multiply(quaternion_azimuth, quaternion_elev)
        angle = get_angle_between_two_quaternions(quaternion_elev_azimuth, initial_quaternion)
        quaternion_final = quaternion_multiply(inverse_of_initial_quaternion, quaternion_elev_azimuth)[0]
        results.append((elev, azimuth, angle, quaternion_final))

    results.sort(key=lambda x: x[2])
    results = [(result[0], result[1], result[3]) for result in results]
    return results


def axis_angle_to_matrix(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Taken from https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))
