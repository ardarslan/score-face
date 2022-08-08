import torch
import numpy as np

import itertools
from pytorch3d.transforms import axis_angle_to_quaternion, matrix_to_quaternion, quaternion_multiply, quaternion_invert
from typing import Dict, Any, OrderedDict, Tuple, List

from rendering import Renderer


def get_quaternion_in_image_and_quaternion_to_make_image_frontal(cfg: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    input_axis_angle_path = cfg["input_axis_angle_path"]
    device = cfg["device"]

    axis_angle = torch.tensor(np.load(input_axis_angle_path), device=device)
    quaternion = axis_angle_to_quaternion(axis_angle)
    quaternion_fix = matrix_to_quaternion(axis_angle_to_matrix(axis="Y", angle=torch.tensor([[np.pi]], device=device)))
    quaternion_to_make_image_frontal = quaternion_multiply(quaternion, quaternion_fix)
    quaternion_in_image = quaternion_invert(quaternion)
    return quaternion_in_image, quaternion_to_make_image_frontal


def get_angle_between_two_quaternions(quaternion_1: torch.Tensor, quaternion_2: torch.Tensor) -> float:
    """
    Taken from https://math.stackexchange.com/a/90098
    """

    dot_product = (quaternion_1 * quaternion_2).sum()
    squared_dot_product = dot_product * dot_product
    cos = 2 * squared_dot_product - 1
    angle = float(torch.arccos(cos))
    return angle


def get_quaternion_to_make_frontal_elev_azimuth(cfg: Dict[str, Any], elev: float, azimuth: float) -> torch.Tensor:
    device = cfg["device"]
    quaternion_to_make_frontal_elev = matrix_to_quaternion(axis_angle_to_matrix(axis="X", angle=torch.tensor([[np.pi * elev / 180.0]], device=device)))
    quaternion_to_make_frontal_azimuth = matrix_to_quaternion(axis_angle_to_matrix(axis="Y", angle=torch.tensor([[np.pi * azimuth / 180.0]], device=device)))
    quaternion_to_make_frontal_elev_azimuth = quaternion_multiply(quaternion_to_make_frontal_elev, quaternion_to_make_frontal_azimuth)
    return quaternion_to_make_frontal_elev_azimuth


def get_quaternion_to_make_image_elev_azimuth(quaternion_to_make_frontal_elev_azimuth: torch.Tensor, quaternion_to_make_image_frontal: torch.Tensor) -> torch.Tensor:
    quaternion_to_make_image_elev_azimuth = quaternion_multiply(quaternion_to_make_image_frontal, quaternion_to_make_frontal_elev_azimuth)[0]
    return quaternion_to_make_image_elev_azimuth


def get_ordered_prerender_results(cfg: Dict[str, Any], elevs: List[float], azimuths: List[float], renderer: Renderer) -> OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]]:
    order_views = cfg["order_views"]

    quaternion_in_image, quaternion_to_make_image_frontal = get_quaternion_in_image_and_quaternion_to_make_image_frontal(cfg=cfg)

    if cfg["initial_view"] == "image":
        initial_quaternion = quaternion_in_image
    elif cfg["initial_view"] == "frontal":
        initial_quaternion = get_quaternion_to_make_frontal_elev_azimuth(cfg=cfg, elev=0.0, azimuth=0.0)
    else:
        raise Exception(f"Not a valid initial_view: {cfg['initial_view']}.")

    elevs_azimuths = list(itertools.product(elevs, azimuths))
    results = []
    for elev, azimuth in elevs_azimuths:
        quaternion_to_make_frontal_elev_azimuth = get_quaternion_to_make_frontal_elev_azimuth(cfg, elev, azimuth)
        quaternion_to_make_image_elev_azimuth = get_quaternion_to_make_image_elev_azimuth(quaternion_to_make_frontal_elev_azimuth=quaternion_to_make_frontal_elev_azimuth, quaternion_to_make_image_frontal=quaternion_to_make_image_frontal)
        pixel_uvs, background_mask = renderer.prerender(quaternion_to_make_image_elev_azimuth)
        angle = get_angle_between_two_quaternions(quaternion_to_make_frontal_elev_azimuth, initial_quaternion)
        results.append((f"{elev}_{azimuth}", angle, (pixel_uvs, background_mask)))
    if order_views.lower() == "true":
        results.sort(key=lambda x: x[1])
    results = OrderedDict([(result[0], result[2]) for result in results])
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
