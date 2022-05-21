import os
import cv2
from tqdm import tqdm
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resizes images in input_images_dir and saves them into output_images_dir. Final image size is (output_images_height, output_images_width).')
    parser.add_argument('--input_images_dir', type=str, default='/cluster/scratch/aarslan/FFHQ/raw', help='Directory for input images.')
    parser.add_argument('--output_images_dir', type=str, default='/cluster/scratch/aarslan/FFHQ/resized', help='Directory for output images.')
    parser.add_argument('--output_images_height', type=int, default=256, help='Height of output images.')
    parser.add_argument('--output_images_width', type=int, default=256, help='Width of output images.')
    args = parser.parse_args()

    os.makedirs(args.output_images_dir, exist_ok=True)

    for image_name in tqdm(sorted(os.listdir(args.input_images_dir))):
        if image_name[-4:] != ".png":
            continue
        input_image_path = os.path.join(args.input_images_dir, image_name)
        output_image_path = os.path.join(args.output_images_dir, image_name)
        input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
        output_image = cv2.resize(
            input_image,
            (args.output_images_height, args.output_images_width),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imwrite(output_image_path, output_image)

        # Uncomment the next line if working with all images.
        # os.system(f"rm -rf {input_image_path}")
