import os
import cv2
import math
import random
import time
import pprint
from uuid import uuid4

import torch
import numpy as np


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_results_dir(image_save_dir, experiment_name):
    results_dir = os.path.join(image_save_dir, experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def save_cfg(cfg):
    results_dir = get_results_dir(image_save_dir=cfg["image_save_dir"], experiment_name=cfg["experiment_name"])
    cfg_path = os.path.join(results_dir, "cfg.txt")
    with open(cfg_path, "w") as cfg_writer:
        cfg_writer.write(pprint.pformat(cfg, indent=4))


def set_experiment_name(cfg):
    cfg["experiment_name"] = f"{int(time.time())}_{uuid4().hex}"


def get_initial_textures(texture_path, _3dmm, small_texture_size, large_texture_size, device):
    texture = cv2.imread(texture_path)
    gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
    if _3dmm == "deca":
        black_mask = gray <= 0
        black_mask_moved = np.vstack((np.zeros((1, black_mask.shape[1])), black_mask[:-1, :]))
        black_mask = np.logical_or(black_mask, black_mask_moved)
    elif _3dmm == "tf_flame":
        black_mask = (gray <= 40)
    else:
        raise Exception(f"Not a valid _3dmm {_3dmm}.")
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


def get_current_small_texture(small_texture_size, large_texture_size, pixel_uvs, small_texture, large_texture):
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
                    small_texture[sample_idx, :, v_small, u_small] = torch.clamp(large_texture[sample_idx, :, v_large, u_large], min=0.0, max=1.0)
    return small_texture


def get_current_large_texture(large_texture_size, small_texture, initial_large_texture, initial_filled_mask):
    upsampled_small_texture = torch.nn.functional.interpolate(input=small_texture, size=large_texture_size, mode='nearest')
    return torch.where(initial_filled_mask, initial_large_texture, upsampled_small_texture)


def set_3dmm_result_paths(cfg):
    if cfg["3dmm"] == "deca":
        cfg["texture_path"] = f"/local/home/aarslan/DECA/TestSamples/examples/results/{cfg['subject_id']}/{cfg['subject_id']}.png"
        cfg["obj_path"] = f"/local/home/aarslan/DECA/TestSamples/examples/results/{cfg['subject_id']}/{cfg['subject_id']}.obj"
        raise Exception("We need to output rotation vector from DECA.")
    elif cfg["3dmm"] == "tf_flame":
        cfg["texture_path"] = f"/local/home/aarslan/TF_FLAME/results/{cfg['subject_id']}.png"
        cfg["obj_path"] = f"/local/home/aarslan/TF_FLAME/results/{cfg['subject_id']}.obj"
        cfg["axis_angle_path"] = f"/local/home/aarslan/TF_FLAME/results/{cfg['subject_id']}_axis_angle.npy"
    else:
        raise Exception(f"Not a valid 3dmm {cfg['3dmm']}.")


def get_target_background(batch_size, num_channels, image_size, device):
    image_shape = (batch_size, num_channels, image_size, image_size)
    target_background = torch.zeros(size=image_shape, device=device)
    return target_background


def get_grad_texture(texture, grad_face, render_func):
    _, grad_texture = torch.autograd.functional.vjp(func=render_func, inputs=texture, v=grad_face, create_graph=False, strict=False)
    return grad_texture


def get_filled_mask(texture, num_channels):
    return (texture.sum(axis=1) != num_channels).unsqueeze(1).repeat(repeats=[1, num_channels, 1, 1])


def load_image_axis_angle(axis_angle_path, device):
    return torch.tensor(np.load(axis_angle_path), device=device)


def axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
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