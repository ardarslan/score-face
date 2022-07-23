import os
import cv2
import random
import time
import pprint
from uuid import uuid4

import torch
import numpy as np


def set_seeds(cfg):
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])


def get_results_dir(cfg):
    results_dir = os.path.join(cfg["image_save_dir"], cfg["experiment_name"])
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def save_cfg(cfg):
    results_dir = get_results_dir(cfg)
    cfg_path = os.path.join(results_dir, "cfg.txt")
    with open(cfg_path, "w") as cfg_writer:
        cfg_writer.write(pprint.pformat(cfg, indent=4))


def get_experiment_name():
    return f"{int(time.time())}_{uuid4().hex}"


def get_initial_texture(cfg):
    initial_texture = cv2.imread(cfg["texture_path"])
    gray = cv2.cvtColor(initial_texture, cv2.COLOR_BGR2GRAY)
    black_mask = gray <= 0
    black_mask_moved = np.vstack((np.zeros((1, black_mask.shape[1])), black_mask[:-1, :]))
    black_mask = np.logical_or(black_mask, black_mask_moved)
    black_mask = np.tile(black_mask[:, :, None], reps=[1, 1, 3])
    initial_texture = np.clip(initial_texture / 255.0 + black_mask, 0, 1)[:, :, [2, 1, 0]]
    initial_texture = cv2.resize(initial_texture, dsize=(cfg["texture_size"], cfg["texture_size"]), interpolation=cv2.INTER_NEAREST)
    initial_texture = torch.tensor(initial_texture, device=cfg["device"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return initial_texture


def get_target_background(cfg):
    image_shape = (cfg["batch_size"], cfg["num_channels"], cfg["image_size"], cfg["image_size"])
    target_background = torch.zeros(size=image_shape, device=cfg["device"]) # black
    # target_background[:, 1, :, :] = target_background[:, 1, :, :] + 1
    return target_background


def update_texture(cfg, current_texture, pixel_uvs, current_optimized_face_mean, unfilled_mask, update_round):
    pixel_uvs[:, :, :, 0] = torch.floor((pixel_uvs[:, :, :, 0] + 1) * cfg["texture_size"] / 2)
    pixel_uvs[:, :, :, 1] = cfg["texture_size"] - 1 - torch.floor(cfg["texture_size"] * (pixel_uvs[:, :, :, 1] + 1) / 2)
    current_optimized_face_mean = torch.clamp(current_optimized_face_mean, min=0.0, max=1.0)

    for sample_idx in range(cfg["batch_size"]):
        if update_round == 0:
            gray = cv2.cvtColor(current_optimized_face_mean[sample_idx, :, :, :].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2GRAY)
        for y in range(cfg["image_size"]):
            for x in range(cfg["image_size"]):
                if unfilled_mask[sample_idx, 0, y, x] == 1:
                    if (update_round == 0 and gray[y, x] > 0.2) or update_round == 1:
                        u = int(pixel_uvs[sample_idx, y, x, 0])
                        v = int(pixel_uvs[sample_idx, y, x, 1])
                        current_texture[sample_idx, :, v, u] = current_optimized_face_mean[sample_idx, :, y, x]
    return current_texture