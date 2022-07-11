import numpy as np
import yaml
import os
from tensorflow.keras.models import load_model
from src.data.ImageDataset import ImageDataset
from src.features.metrics import compute_auc_metrics
from src.models.log.LogTracker import LogTracker
from datetime import datetime

from definitions import MODELS_PATH, MENPO_2D_DICT_PATH, MENPO_2D_IMAGES_LIST_PATH, MENPO_2D_LANDMARKS_LIST_PATH, \
    get_menpo_paths, SECRETS_PATH


def load_module(path, module):
    return load_model(os.path.join(path, f"{module}_model.h5"), compile=False)


SELECTED_MODEL = "model_TFM-40_2022-07-06_16.22.15"

if __name__ == "__main__":
    with open(SECRETS_PATH, 'r') as file:
        secrets = yaml.safe_load(file)

    dict_menpo_2d = np.load(MENPO_2D_DICT_PATH, allow_pickle=True).item()
    images_list = np.load(MENPO_2D_IMAGES_LIST_PATH, allow_pickle=True)
    landmarks_list = np.load(MENPO_2D_LANDMARKS_LIST_PATH, allow_pickle=True)

    test_sample_path, test_target_path = get_menpo_paths("train", include_target=True)

    test_dataset = ImageDataset(test_sample_path, test_target_path, batch_size=1, shuffle=False, tag="train")

    run_id = SELECTED_MODEL.split("_")[1]
    model_dir_path = os.path.join(MODELS_PATH, SELECTED_MODEL)

    neptune_manager = LogTracker(secrets["neptune"], run_id=run_id)

    generator = load_module(model_dir_path, "generator")
    discriminator = load_module(model_dir_path, "discriminator")

    start_time = datetime.now()
    auc, auc_plot, nppe_max, nppe_min, nppe_std, nppe_mean = compute_auc_metrics(generator, test_dataset, dict_menpo_2d,
                                                                                 threshold=0.75)
    end_time = datetime.now()

    neptune_manager.save_image("evaluation/auc-plot", auc_plot)
    neptune_manager.save_static_variable("evaluation/auc", auc)
    neptune_manager.save_static_variable("evaluation/nppe-max", nppe_max)
    neptune_manager.save_static_variable("evaluation/nppe-min", nppe_min)
    neptune_manager.save_static_variable("evaluation/nppe-std", nppe_std)
    neptune_manager.save_static_variable("evaluation/nppe-mean", nppe_mean)

    print(f"Evaluation finished. Elapsed time: {end_time - start_time}")
