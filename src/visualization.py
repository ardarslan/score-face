import os
import cv2
import torch
import numpy as np
from PIL import Image
from utils import get_experiment_dir
from rendering import Renderer
from typing import Dict, Any, List, Tuple, OrderedDict


def save_images(cfg: Dict[str, Any], _images: torch.Tensor, image_type: str, iteration: int, dark_pixel_allowed: bool, elev: float, azimuth: float) -> None:
    images = (np.clip(_images.permute(0, 2, 3, 1).detach().cpu().numpy(), a_min=0.0, a_max=1.0) * 255.0).astype(np.uint8)
    experiment_dir = get_experiment_dir(cfg=cfg)
    image_main_folder_path = os.path.join(experiment_dir, "intermediates", image_type, f"darkpix_{dark_pixel_allowed}_elev_{elev}_azimuth_{azimuth}")
    os.makedirs(image_main_folder_path, exist_ok=True)
    for image_index, image in enumerate(images):
        image_iteration_folder_path = os.path.join(image_main_folder_path, str(iteration).zfill(6))
        os.makedirs(image_iteration_folder_path, exist_ok=True)
        image_file_path = os.path.join(image_iteration_folder_path, str(image_index).zfill(6) + ".png")
        image = image[:, :, [2, 1, 0]]
        cv2.imwrite(image_file_path, image)


def save_optimization_gif(cfg: Dict[str, Any], dark_pixel_allowed: bool, elev: float, azimuth: float) -> None:
    experiment_dir = get_experiment_dir(cfg=cfg)
    face_main_folder_path = os.path.join(experiment_dir, "intermediates", "face", f"darkpix_{dark_pixel_allowed}_elev_{elev}_azimuth_{azimuth}")
    face_iteration_folder_paths = [os.path.join(face_main_folder_path, iteration) for iteration in sorted(os.listdir(face_main_folder_path))]
    face_iteration_folder_paths = [face_iteration_folder_path for face_iteration_folder_path in face_iteration_folder_paths if os.path.isdir(face_iteration_folder_path)]
    face_file_paths = [os.path.join(face_iteration_folder_path, "0".zfill(6) + ".png") for face_iteration_folder_path in face_iteration_folder_paths]
    gif_file_path = os.path.join(face_main_folder_path, "animation.gif")
    imgs = (Image.open(face_file_path) for face_file_path in face_file_paths)
    img = next(imgs)
    img.save(fp=gif_file_path, format='GIF', append_images=imgs,
             save_all=True, duration=100, loop=0)


def save_outputs(cfg: Dict[str, Any], final_texture: torch.Tensor, target_background: torch.Tensor, renderer: Renderer, elevs: List[float], azimuths: List[float], ordered_prerender_results: OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]]) -> None:
    step_azimuth = cfg["step_azimuth"]
    step_elev = cfg["step_elev"]

    experiment_dir = get_experiment_dir(cfg=cfg)
    outputs_dir = os.path.join(experiment_dir, "outputs")
    face_outputs_dir = os.path.join(outputs_dir, "face")
    texture_outputs_dir = os.path.join(outputs_dir, "texture")
    os.makedirs(face_outputs_dir, exist_ok=True)
    os.makedirs(texture_outputs_dir, exist_ok=True)

    texture_file_path = os.path.join(texture_outputs_dir, str(0).zfill(6) + ".png")
    texture_np = (final_texture[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)[:, :, [2, 1, 0]]
    cv2.imwrite(texture_file_path, texture_np)

    min_elev, max_elev = np.min(elevs), np.max(elevs)
    min_azimuth, max_azimuth = np.min(azimuths), np.max(azimuths)

    elevs_azimuths = []
    
    current_elev = min_elev
    current_azimuth = min_azimuth

    while current_azimuth <= max_azimuth:
        elevs_azimuths.append((current_elev, current_azimuth))
        current_azimuth += step_azimuth
    current_azimuth = max_azimuth

    while current_elev <= max_elev:
        elevs_azimuths.append((current_elev, current_azimuth))
        current_elev += step_elev
    current_elev = max_elev

    while current_azimuth >= min_azimuth:
        elevs_azimuths.append((current_elev, current_azimuth))
        current_azimuth -= step_azimuth
    current_azimuth = min_azimuth
    
    while current_elev >= min_elev:
        elevs_azimuths.append((current_elev, current_azimuth))
        current_elev -= step_elev
    current_elev = min_elev

    for index, (elev, azimuth) in enumerate(elevs_azimuths):
        pixel_uvs, background_mask = ordered_prerender_results[f"{elev}_{azimuth}"]
        current_face = renderer.render(texture=final_texture[0].unsqueeze(0), background=target_background[0].unsqueeze(0), pixel_uvs=pixel_uvs, background_mask=background_mask)
        current_face = (current_face[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)[:, :, [2, 1, 0]]
        image_file_path = os.path.join(face_outputs_dir, f"{str(index).zfill(6)}_{elev}_{azimuth}.png")
        cv2.imwrite(image_file_path, current_face)
    
    face_file_paths = [os.path.join(face_outputs_dir, file_name) for file_name in sorted(os.listdir(face_outputs_dir))]
    gif_file_path = os.path.join(outputs_dir, "animation.gif")
    imgs = (Image.open(face_file_path) for face_file_path in face_file_paths)
    img = next(imgs)
    img.save(fp=gif_file_path, format='GIF', append_images=imgs,
             save_all=True, duration=100, loop=0)
