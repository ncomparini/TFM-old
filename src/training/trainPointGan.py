import numpy as np

from src.data.PointDataset import PointDataset
from src.models.gan.PointGan import PointGan
from src.models.gan.discriminator.PointDiscriminator import PointDiscriminator
from src.models.gan.GenerativeAdversarialNetwork import GenerativeAdversarialNetwork
from src.models.gan.generator.Image2PointGenerator import Image2PointGenerator
from src.models.log.LogTracker import LogTracker
from definitions import CONFIG_PATH, SECRETS_PATH, get_menpo_paths, MENPO_2D_DICT_PATH
import yaml

MODEL_TYPE = "point-gan"

if __name__ == "__main__":
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)

    with open(SECRETS_PATH, 'r') as file:
        secrets = yaml.safe_load(file)

    points_dictionary = np.load(MENPO_2D_DICT_PATH, allow_pickle=True).item()

    train_sample_path = get_menpo_paths("train", False)
    test_sample_path = get_menpo_paths("test", False)

    config_gan = config["hyperparameters"]["gan"][MODEL_TYPE]
    batch_size = config_gan["batch-size"]
    number_of_landmarks = config_gan["number-of-landmarks"]

    train_dataset = PointDataset(train_sample_path, points_dictionary, batch_size,
                                 shuffle=True, tag="train", number_of_landmarks=number_of_landmarks)
    test_dataset = PointDataset(test_sample_path, points_dictionary, batch_size,
                                shuffle=False, tag="test", number_of_landmarks=number_of_landmarks)

    generator = Image2PointGenerator(config_gan)
    discriminator = PointDiscriminator(config_gan, patch_gan=True)
    neptune_manager = LogTracker(secrets["neptune"])

    gan = PointGan(generator, discriminator, config_gan,
                   log_tracker=neptune_manager,
                   allow_disc_switch_off=False)

    gan.fit(train_dataset, config_gan["epochs"], test_dataset)
