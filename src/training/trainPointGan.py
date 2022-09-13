import numpy as np

from src.data.PointDataset import PointDataset
from src.models.gan.PointGan import PointGan
from src.models.gan.discriminator.PointDiscriminator import PointDiscriminator
from src.models.gan.generator.Image2PointGenerator import Image2PointGenerator
from src.models.log.LogTracker import LogTracker
from definitions import CONFIG_PATH, SECRETS_PATH, get_menpo_paths, get_dict_path
import yaml

MODEL_TYPE = "point-gan"

if __name__ == "__main__":
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)

    with open(SECRETS_PATH, 'r') as file:
        secrets = yaml.safe_load(file)

    type_config = config["gan"]["image-to-point"]
    gan_config = type_config[MODEL_TYPE]
    data_config = type_config["data"]

    points_dictionary = np.load(get_dict_path(data_config), allow_pickle=True).item()

    train_sample_path = get_menpo_paths(data_config, "train", include_image_target=False)
    test_sample_path = get_menpo_paths(data_config, "test", include_image_target=False)

    params_config = gan_config["hyperparameters"]
    batch_size = params_config["batch-size"]
    number_of_landmarks = params_config["number-of-landmarks"]

    train_dataset = PointDataset(train_sample_path, points_dictionary, batch_size,
                                 shuffle=True, tag="train", number_of_landmarks=number_of_landmarks)
    test_dataset = PointDataset(test_sample_path, points_dictionary, batch_size,
                                shuffle=False, tag="test", number_of_landmarks=number_of_landmarks)

    generator = Image2PointGenerator(params_config)
    discriminator = PointDiscriminator(params_config, patch_gan=False)
    neptune_manager = LogTracker(secrets["neptune"])

    gan = PointGan(generator, discriminator, params_config,
                   log_tracker=neptune_manager,
                   allow_disc_switch_off=False)

    gan.fit(train_dataset, params_config["epochs"], test_dataset)
