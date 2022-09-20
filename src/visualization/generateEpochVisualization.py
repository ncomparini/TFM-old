import yaml
import os
import src.features.viztools as viz
from src.models.log.LogTracker import LogTracker

from definitions import MODELS_PATH, SECRETS_PATH, INTERIM_DATA_DIR

SELECTED_MODEL = "model_TFM-93_2022-09-18_00.25.32"

if __name__ == "__main__":
    with open(SECRETS_PATH, 'r') as file:
        secrets = yaml.safe_load(file)

    run_id = SELECTED_MODEL.split("_")[1]
    model_dir_path = os.path.join(MODELS_PATH, SELECTED_MODEL)

    neptune_manager = LogTracker(secrets["neptune"], run_id=run_id)

    results_path = os.path.join(INTERIM_DATA_DIR, SELECTED_MODEL)
    sample_dir = os.path.join(results_path, "sample")
    target_dir = os.path.join(results_path, "target")
    epochs_dir = os.path.join(results_path, "epochs")
    partition = "test"

    neptune_manager.download_predicted_epoch_images(partition, epochs_dir)
    neptune_manager.download_original_epoch_images(partition, "sample", results_path)
    neptune_manager.download_original_epoch_images(partition, "target", results_path)
    viz.generate_gif(sample_dir, target_dir, epochs_dir, results_path)
