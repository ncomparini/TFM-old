import tensorflow as tf
import glob
import os
from definitions import ROOT_DIR


class ImageDataset:

    def __init__(self, path_sample_images, path_target_images, batch_size, shuffle, tag):
        self.path_sample_images = path_sample_images
        self.path_target_images = path_target_images
        self.batch_size = batch_size
        self.tag = tag

        self.sample_images_urls = self._read_image_urls(self.path_sample_images)
        self.target_images_urls = self._read_image_urls(self.path_target_images)

        self.dataset = self._create_dataset(shuffle)

    def _read_image_urls(self, path):
        urls = glob.glob(os.path.join(ROOT_DIR, path, "*"))
        sorted_urls = sorted(urls, key=lambda url: int(os.path.basename(url).split(".")[0].split("_")[0]))

        return sorted_urls

    def _load(self, img_file, type):
        img = tf.io.read_file(img_file)

        if type == "jpg":
            img = tf.image.decode_jpeg(img, channels=3)
        elif type == "png":
            img = tf.image.decode_png(img, channels=1)
        else:
            return None

        img = tf.cast(img, tf.float32)
        return img

    def _normalize(self, image):
        return (image / 127.5) - 1

    def _load_paired_image(self, input_img_file, output_img_file):
        input_img = self._load(input_img_file, type="jpg")
        norm_input_img = self._normalize(input_img)

        output_img = self._load(output_img_file, type="png")
        norm_output_img = self._normalize(output_img)

        return norm_input_img, norm_output_img

    def _create_dataset(self, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices((self.sample_images_urls, self.target_images_urls))
        dataset = dataset.map(self._load_paired_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(len(self.sample_images_urls), reshuffle_each_iteration=False)
        dataset = dataset.batch(self.batch_size)

        return dataset
