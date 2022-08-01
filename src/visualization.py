import os
import cv2
import numpy as np
from PIL import Image
from utils import get_results_dir


def save_images(_images, image_type, iteration, elev, azimuth, image_save_dir, experiment_name, saved_image_size):
    images = (np.clip(_images.permute(0, 2, 3, 1).detach().cpu().numpy(), a_min=0.0, a_max=1.0) * 255.0).astype(np.uint8)
    results_dir = get_results_dir(image_save_dir=image_save_dir, experiment_name=experiment_name)
    image_main_folder_path = os.path.join(results_dir, "intermediates", image_type, f"elev_{elev}_azimuth_{azimuth}")
    os.makedirs(image_main_folder_path, exist_ok=True)
    for image_index, image in enumerate(images):
        image_iteration_folder_path = os.path.join(image_main_folder_path, str(iteration).zfill(6))
        os.makedirs(image_iteration_folder_path, exist_ok=True)
        image_file_path = os.path.join(image_iteration_folder_path, str(image_index).zfill(6) + ".png")
        image = image[:, :, [2, 1, 0]]
        if image_type == "face":
            image = cv2.resize(image, dsize=(saved_image_size, saved_image_size))
        cv2.imwrite(image_file_path, image)


def save_optimization_gif(elev, azimuth, image_save_dir, experiment_name):
    results_dir = get_results_dir(image_save_dir=image_save_dir, experiment_name=experiment_name)
    face_main_folder_path = os.path.join(results_dir, "intermediates", "face", f"elev_{elev}_azimuth_{azimuth}")
    face_iteration_folder_paths = [os.path.join(face_main_folder_path, iteration) for iteration in sorted(os.listdir(face_main_folder_path))]
    face_iteration_folder_paths = [face_iteration_folder_path for face_iteration_folder_path in face_iteration_folder_paths if os.path.isdir(face_iteration_folder_path)]
    face_file_paths = [os.path.join(face_iteration_folder_path, "0".zfill(6) + ".png") for face_iteration_folder_path in face_iteration_folder_paths] # Create gif for only first sample in the batch
    gif_file_path = os.path.join(face_main_folder_path, "animation.gif")
    imgs = (Image.open(face_file_path) for face_file_path in face_file_paths)
    img = next(imgs)
    img.save(fp=gif_file_path, format='GIF', append_images=imgs,
             save_all=True, duration=100, loop=0)


def save_outputs(large_texture, small_texture, target_background, renderer, elevs, azimuths, image_save_dir, experiment_name, obj_path, axis_angle_path):
    results_dir = get_results_dir(image_save_dir=image_save_dir, experiment_name=experiment_name)
    outputs_dir = os.path.join(results_dir, "outputs")
    face_outputs_dir = os.path.join(outputs_dir, "face")
    large_texture_outputs_dir = os.path.join(outputs_dir, "large_texture")
    small_texture_outputs_dir = os.path.join(outputs_dir, "small_texture")
    os.makedirs(face_outputs_dir, exist_ok=True)
    os.makedirs(large_texture_outputs_dir, exist_ok=True)
    os.makedirs(small_texture_outputs_dir, exist_ok=True)

    large_texture_np = (large_texture[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    large_texture_file_path = os.path.join(large_texture_outputs_dir, str(0).zfill(6) + ".png")
    large_texture_np = large_texture_np[:, :, [2, 1, 0]]
    cv2.imwrite(large_texture_file_path, large_texture_np)

    small_texture_np = (small_texture[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    small_texture_file_path = os.path.join(small_texture_outputs_dir, str(0).zfill(6) + ".png")
    small_texture_np = small_texture_np[:, :, [2, 1, 0]]
    cv2.imwrite(small_texture_file_path, small_texture_np)

    min_elev, max_elev = np.min(elevs), np.max(elevs)
    min_azimuth, max_azimuth = np.min(azimuths), np.max(azimuths)

    elevs_azimuths = []
    
    current_elev = min_elev
    current_azimuth = min_azimuth

    while current_azimuth <= max_azimuth:
        elevs_azimuths.append((current_elev, current_azimuth))
        current_azimuth += 1.0
    current_azimuth = max_azimuth

    while current_elev <= max_elev:
        elevs_azimuths.append((current_elev, current_azimuth))
        current_elev += 1.0
    current_elev = max_elev

    while current_azimuth >= min_azimuth:
        elevs_azimuths.append((current_elev, current_azimuth))
        current_azimuth -= 1.0
    current_azimuth = min_azimuth
    
    while current_elev >= min_elev:
        elevs_azimuths.append((current_elev, current_azimuth))
        current_elev -= 1.0
    current_elev = min_elev

    for index, (elev, azimuth) in enumerate(elevs_azimuths):
        image_file_path = os.path.join(face_outputs_dir, f"{str(index).zfill(6)}_{elev}_{azimuth}.png")
        pixel_uvs, background_mask = renderer.prerender(obj_path=obj_path, axis_angle_path=axis_angle_path, texture=large_texture, elev=elev, azimuth=azimuth)
        current_face = renderer.render(texture=large_texture[0].unsqueeze(0), background=target_background[0].unsqueeze(0), pixel_uvs=pixel_uvs, background_mask=background_mask)
        current_face = (current_face[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)[:, :, [2, 1, 0]]
        cv2.imwrite(image_file_path, current_face)
    
    face_file_paths = [os.path.join(face_outputs_dir, file_name) for file_name in sorted(os.listdir(face_outputs_dir))]
    gif_file_path = os.path.join(outputs_dir, "animation.gif")
    imgs = (Image.open(face_file_path) for face_file_path in face_file_paths)
    img = next(imgs)
    img.save(fp=gif_file_path, format='GIF', append_images=imgs,
             save_all=True, duration=100, loop=0)
