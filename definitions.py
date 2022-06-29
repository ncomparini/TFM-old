import os

SEED = 1024
ROOT_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(ROOT_DIR, "config.yml")
SECRETS_PATH = os.path.join(ROOT_DIR, "secrets.yml")
MODELS_PATH = os.path.join(ROOT_DIR, "models")

DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MENPO_2D_DIR = os.path.join(PROCESSED_DATA_DIR, "menpo-2d-frontals")
MENPO_2D_DICT_PATH = os.path.join(MENPO_2D_DIR, "dictMenpo2D.npy")
MENPO_2D_IMAGES_LIST_PATH = os.path.join(MENPO_2D_DIR, "imgslist.npy")
MENPO_2D_LANDMARKS_LIST_PATH = os.path.join(MENPO_2D_DIR, "landmarkslist.npy")


def get_menpo_paths(partition: str, include_target: bool = True):
    partition_dir = os.path.join(MENPO_2D_DIR, partition)
    if include_target:
        return os.path.join(partition_dir, "input"), os.path.join(partition_dir, "output")
    else:
        return os.path.join(partition_dir, "input")
