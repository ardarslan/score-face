import os
import random
import argparse
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


def get_argument_parser():
    parser = argparse.ArgumentParser(description="Arguments for running the script")
    parser.add_argument(
        "--checkpoint_filepath", type=str, default="../exp/ve/ffhq_256_ncsnpp_continuous/checkpoint_48.pth"
    )
    parser.add_argument(
        "--image_save_dir", type=str, default="../images"
    )
    parser.add_argument(
        "--obj_path", type=str, default="../assets/40044.obj"
    )
    parser.add_argument(
        "--image_size", type=int, default=256
    )
    parser.add_argument(
        "--texture_size", type=int, default=256
    )
    parser.add_argument(
        "--num_channels", type=int, default=3
    )
    parser.add_argument(
        "--batch_size", type=int, default=1
    )
    parser.add_argument(
        "--predict", type=str, default="never", choices=["never", "face_always", "face_once", "texture_and_background"]
    )
    parser.add_argument(
        "--num_corrector_steps", type=int, default=6
    )
    parser.add_argument(
        "--sampling_eps", type=float, default=1e-5
    )
    parser.add_argument(
        "--probability_flow", type=bool, default=False
    )
    parser.add_argument(
        "--render_mode", type=str, default="3d_fixed_background", choices=["background", "2d", "3d", "2d_modified", "3d_fixed_background"]
    )
    parser.add_argument(
        "--cuda_device", type=int, default=0
    )
    parser.add_argument(
        "--elev_azimuth_random", type=bool, default=False
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--sde_T", type=int, default=1 # 1 
    )
    parser.add_argument(
        "--sde_N", type=int, default=1000 # 2000 
    )
    # parser.add_argument(
    #     "--add_noise", type=str, default="texture_and_background", choices=["never", "face", "texture_and_background"]
    # )
    # parser.add_argument(
    #     "--texture_snr", type=float, default=0.05
    # )
    # parser.add_argument(
    #     "--background_snr", type=float, default=0.05
    # )
    # parser.add_argument(
    #     "--texture_initial_noise_coefficient", type=float, default=300
    # )
    # parser.add_argument(
    #     "--background_initial_noise_coefficient", type=float, default=300
    # )
    # parser.add_argument(
    #     "--texture_noise_coefficient", type=float, default=300
    # )
    # parser.add_argument(
    #     "--background_noise_coefficient", type=float, default=300
    # )
    return parser
