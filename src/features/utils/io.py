import tensorflow as tf
import glob
import os
from definitions import ROOT_DIR


def read_image_urls(path):
    urls = glob.glob(os.path.join(ROOT_DIR, path, "*"))
    sorted_urls = sorted(urls, key=lambda url: int(os.path.basename(url).split(".")[0].split("_")[0]))

    return sorted_urls


def load_image(img_file, img_type):
    img = tf.io.read_file(img_file)

    if img_type == "jpg":
        img = tf.image.decode_jpeg(img, channels=3)
    elif img_type == "png":
        img = tf.image.decode_png(img, channels=1)
    else:
        return None

    img = tf.cast(img, tf.float32)
    return img


def load_module(path, module):
    return tf.keras.models.load_model(os.path.join(path, f"{module}_model.h5"), compile=False)


def save_model_diagram(path, model):
    return tf.keras.utils.plot_model(model, to_file=path, show_shapes=True)
