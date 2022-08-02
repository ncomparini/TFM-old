import os

from definitions import MODELS_PATH
from src.features.utils.io import load_module, save_model_diagram


if __name__ == "__main__":
    for selected_model in os.listdir(MODELS_PATH):
        model_dir_path = os.path.join(MODELS_PATH, selected_model)

        generator = load_module(model_dir_path, "generator")
        discriminator = load_module(model_dir_path, "discriminator")

        save_model_diagram(os.path.join(model_dir_path, "generator.png"), generator)
        save_model_diagram(os.path.join(model_dir_path, "discriminator.png"), discriminator)
