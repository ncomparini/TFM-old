import numpy as np
import yaml
import os
from src.data.PointDataset import PointDataset
from src.features.metrics import compute_auc_metrics
from src.features.utils.io import load_module
from src.models.log.LogTracker import LogTracker
from datetime import datetime

from definitions import MODELS_PATH, MENPO_2D_DICT_PATH, get_menpo_paths, SECRETS_PATH

SELECTED_MODEL = "model_TFM-64_2022-07-14_00.59.39"

if __name__ == "__main__":
    with open(SECRETS_PATH, 'r') as file:
        secrets = yaml.safe_load(file)

    dict_menpo_2d = np.load(MENPO_2D_DICT_PATH, allow_pickle=True).item()

    test_sample_path = get_menpo_paths("train", include_image_target=False)

    test_dataset = PointDataset(test_sample_path, dict_menpo_2d, batch_size=1, shuffle=False, tag="train",
                                number_of_landmarks=68)

    run_id = SELECTED_MODEL.split("_")[1]
    model_dir_path = os.path.join(MODELS_PATH, SELECTED_MODEL)

    neptune_manager = LogTracker(secrets["neptune"], run_id=run_id)

    generator = load_module(model_dir_path, "generator")
    discriminator = load_module(model_dir_path, "discriminator")

    start_time = datetime.now()
    auc, auc_plot, nppe_max, nppe_min, nppe_std, nppe_mean = compute_auc_metrics(generator, test_dataset, dict_menpo_2d,
                                                                                 is_image_output=False,
                                                                                 number_of_landmarks=68)
    end_time = datetime.now()

    neptune_manager.save_image("evaluation/auc-plot", auc_plot)
    neptune_manager.save_static_variable("evaluation/auc", auc)
    neptune_manager.save_static_variable("evaluation/nppe-max", nppe_max)
    neptune_manager.save_static_variable("evaluation/nppe-min", nppe_min)
    neptune_manager.save_static_variable("evaluation/nppe-std", nppe_std)
    neptune_manager.save_static_variable("evaluation/nppe-mean", nppe_mean)

    print(f"Evaluation finished. Elapsed time: {end_time - start_time}")
