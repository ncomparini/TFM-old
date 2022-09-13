import os

from definitions import MODELS_PATH
from src.features.utils.io import load_module, save_model_diagram

if __name__ == "__main__":
    for selected_model in os.listdir(MODELS_PATH):
        model_dir_path = os.path.join(MODELS_PATH, selected_model)
        save_path_generator = os.path.join(model_dir_path, "generator.png")
        save_path_discriminator = os.path.join(model_dir_path, "discriminator.png")

        if not os.path.exists(save_path_generator):
            generator = load_module(model_dir_path, "generator")
            save_model_diagram(save_path_generator, generator)
            print(f"Generator diagram saved for model: {selected_model}")

        if not os.path.exists(save_path_discriminator):
            discriminator = load_module(model_dir_path, "discriminator")
            save_model_diagram(save_path_discriminator, discriminator)
            print(f"Discriminator diagram saved for model: {selected_model}")
