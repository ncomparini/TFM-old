import os

SEED = 1024
ROOT_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(ROOT_DIR, "config.yml")
SECRETS_PATH = os.path.join(ROOT_DIR, "secrets.yml")
MODELS_PATH = os.path.join(ROOT_DIR, "models")

DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")


def get_data_dir(config):
    return os.path.join(PROCESSED_DATA_DIR, config["data-dir"])


def get_dict_path(config):
    return os.path.join(PROCESSED_DATA_DIR, config["data-dir"], "dictMenpo2D.npy")


def get_images_list_path(config):
    return os.path.join(PROCESSED_DATA_DIR, config["data-dir"], "imgslist.npy")


def get_landmarks_list_path(config):
    return os.path.join(PROCESSED_DATA_DIR, config["data-dir"], "landmarkslist.npy")


def get_menpo_paths(config, partition: str, include_image_target: bool = True):
    partition_dir = os.path.join(get_data_dir(config), partition)
    if include_image_target:
        return os.path.join(partition_dir, "input"), os.path.join(partition_dir, "output")
    else:
        return os.path.join(partition_dir, "input")
