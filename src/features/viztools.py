import os

import imageio.v3 as iio
import imageio
from pathlib import Path

import numpy as np


def read_images_concatenated(images_dir, max_images=3, to_rgb=False):
    images = list()

    for num_img in range(max_images):
        images.append(iio.imread(os.path.join(images_dir, f"{num_img}.png")))

    concatenated_images = np.concatenate(images, axis=1)
    if to_rgb:
        return np.stack((concatenated_images,) * 3, axis=-1)

    return concatenated_images


def generate_gif(sample_dir, target_dir, epoch_images_dir, destination_dir, repetition=3):
    epoch_images = list()
    gif_save_path = os.path.join(destination_dir, "results.gif")

    sample_image = read_images_concatenated(sample_dir)
    target_image = read_images_concatenated(target_dir, to_rgb=True)

    for folder in Path(epoch_images_dir).iterdir():
        if not folder.is_file():
            epoch_image = read_images_concatenated(folder, to_rgb=True)
            result_image = np.concatenate([sample_image, target_image, epoch_image], axis=0)

            for rep in range(repetition):
                epoch_images.append(result_image)

    imageio.mimsave(gif_save_path, epoch_images)
