import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from utils import get_results_dir


def image_grid(cfg, images):
    size = cfg["image_size"]
    channels = cfg["num_channels"]
    images = images.reshape(-1, size, size, channels)
    w = int(np.sqrt(images.shape[0]))
    images = images.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
    return images


def save_images(cfg, images, image_type, iteration):
    images = np.clip(images.permute(0, 2, 3, 1).detach().cpu().numpy(), a_min=0.0, a_max=1.0)
    images = image_grid(cfg, images)
    fig = plt.figure(figsize=(3, 3))
    plt.axis("off")
    plt.imshow(images)
    results_dir = get_results_dir(cfg)
    image_file_path = os.path.join(results_dir, image_type, str(iteration).zfill(6) + ".png")
    os.makedirs(Path(image_file_path).parent.__str__(), exist_ok=True)
    fig.savefig(image_file_path)
    plt.close(fig)