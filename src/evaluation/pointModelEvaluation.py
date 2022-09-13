import numpy as np
import yaml
import os
from src.data.PointDataset import PointDataset
from src.features.metrics import compute_and_save_metrics
from src.features.utils.io import load_module
from src.models.log.LogTracker import LogTracker

from definitions import MODELS_PATH, get_menpo_paths, SECRETS_PATH, CONFIG_PATH, get_dict_path

SELECTED_MODEL = "model_TFM-88_2022-09-12_21.23.18"

if __name__ == "__main__":
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)

    with open(SECRETS_PATH, 'r') as file:
        secrets = yaml.safe_load(file)

    type_config = config["gan"]["image-to-point"]
    data_config = type_config["data"]

    dict_menpo_2d = np.load(get_dict_path(data_config), allow_pickle=True).item()

    train_sample_path = get_menpo_paths(data_config, "train", include_image_target=False)
    test_sample_path = get_menpo_paths(data_config, "test", include_image_target=False)

    train_dataset = PointDataset(train_sample_path, dict_menpo_2d, batch_size=1, shuffle=False,
                                 tag="train", number_of_landmarks=68)
    test_dataset = PointDataset(test_sample_path, dict_menpo_2d, batch_size=1, shuffle=False,
                                tag="test", number_of_landmarks=68)

    run_id = SELECTED_MODEL.split("_")[1]
    model_dir_path = os.path.join(MODELS_PATH, SELECTED_MODEL)

    neptune_manager = LogTracker(secrets["neptune"], run_id=run_id)

    generator = load_module(model_dir_path, "generator")
    discriminator = load_module(model_dir_path, "discriminator")

    compute_and_save_metrics(generator, train_dataset, dict_menpo_2d, neptune_manager, is_image_output=False)
    compute_and_save_metrics(generator, test_dataset, dict_menpo_2d, neptune_manager, is_image_output=False)
