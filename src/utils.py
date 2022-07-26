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


def set_experiment_name(cfg):
    cfg["experiment_name"] = f"{int(time.time())}_{uuid4().hex}"


def get_initial_texture(cfg):
    initial_texture = cv2.imread(cfg["texture_path"])
    gray = cv2.cvtColor(initial_texture, cv2.COLOR_BGR2GRAY)
    
    if cfg["3dmm"] == "deca":
        black_mask = gray <= 0
        black_mask_moved = np.vstack((np.zeros((1, black_mask.shape[1])), black_mask[:-1, :]))
        black_mask = np.logical_or(black_mask, black_mask_moved)
    elif cfg["3dmm"] == "tf_flame":
        black_mask = (gray <= 40)

    black_mask = np.tile(black_mask[:, :, None], reps=[1, 1, 3])
    initial_texture = np.clip(initial_texture / 255 + black_mask, 0, 1)
    initial_texture = initial_texture[:, :, [2, 1, 0]]
    initial_texture = cv2.resize(initial_texture, dsize=(cfg["texture_size"], cfg["texture_size"]), interpolation=cv2.INTER_NEAREST)
    initial_texture = torch.tensor(initial_texture, device=cfg["device"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    return initial_texture


def set_texture_and_obj_path(cfg):
    if cfg["3dmm"] == "deca":
        cfg["texture_path"] = f"/local/home/aarslan/DECA/TestSamples/examples/results/{cfg['subject_id']}/{cfg['subject_id']}.png"
        cfg["obj_path"] = f"/local/home/aarslan/DECA/TestSamples/examples/results/{cfg['subject_id']}/{cfg['subject_id']}.obj"
    elif cfg["3dmm"] == "tf_flame":
        cfg["texture_path"] = "/local/home/aarslan/TF_FLAME/results/33673.png"
        cfg["obj_path"] = "/local/home/aarslan/TF_FLAME/results/33673.obj"
    else:
        raise Exception(f"Not a valid 3dmm {cfg['3dmm']}.")


def get_target_background(cfg):
    image_shape = (cfg["batch_size"], cfg["num_channels"], cfg["image_size"], cfg["image_size"])
    target_background = torch.zeros(size=image_shape, device=cfg["device"])
    return target_background


def update_texture(cfg, current_texture, pixel_uvs, current_optimized_face_mean, unfilled_mask, update_round):
    pixel_uvs[:, :, :, 0] = torch.floor((pixel_uvs[:, :, :, 0] + 1) * cfg["texture_size"] / 2)
    pixel_uvs[:, :, :, 1] = cfg["texture_size"] - 1 - torch.floor(cfg["texture_size"] * (pixel_uvs[:, :, :, 1] + 1) / 2)
    current_optimized_face_mean = torch.clamp(current_optimized_face_mean, min=0.0, max=1.0)

    for sample_idx in range(cfg["batch_size"]):
        # if update_round == 0:
        #   gray = cv2.cvtColor(current_optimized_face_mean[sample_idx, :, :, :].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2GRAY)
        for y in range(cfg["image_size"]):
            for x in range(cfg["image_size"]):
                if unfilled_mask[sample_idx, 0, y, x] == 1:
                    # if (update_round == 0 and gray[y, x] > 0.2) or update_round == 1:
                    current_pixel = current_optimized_face_mean[sample_idx, :, y, x]
                    # if (update_round == 0 and not (current_pixel[1] > 0.8 and current_pixel[0] < 0.2 and current_pixel[2] < 0.2)) or (update_round == 1):
                    u = int(pixel_uvs[sample_idx, y, x, 0])
                    v = int(pixel_uvs[sample_idx, y, x, 1])
                    current_texture[sample_idx, :, v, u] = current_pixel
    return current_texture
