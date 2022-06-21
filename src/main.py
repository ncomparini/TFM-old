from src.models.gan.Generator import Generator
from src.models.gan.Discriminator import Discriminator
from src.models.gan.GenerativeAdversarialNetwork import GenerativeAdversarialNetwork
from src.models.log.LogTracker import LogTracker
from src.data.ImageDataset import ImageDataset
from definitions import CONFIG_PATH, SECRETS_PATH
import yaml
import os

if __name__ == "__main__":
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)

    with open(SECRETS_PATH, 'r') as file:
        secrets = yaml.safe_load(file)

    train_sample_path = config["data"]["menpo"]["train"]["sample"]
    train_target_path = config["data"]["menpo"]["train"]["target"]

    test_sample_path = config["data"]["menpo"]["test"]["sample"]
    test_target_path = config["data"]["menpo"]["test"]["target"]

    config_gan = config["hyperparameters"]["gan"]["patch-gan"]
    batch_size = config_gan["batch-size"]

    train_dataset = ImageDataset(train_sample_path, train_target_path, batch_size, shuffle=True, tag="train")
    test_dataset = ImageDataset(test_sample_path, test_target_path, batch_size, shuffle=False, tag="test")

    generator = Generator(config_gan)
    discriminator = Discriminator(config_gan, patch_gan=True)
    neptune_manager = LogTracker(secrets["neptune"], "FLE")

    gan = GenerativeAdversarialNetwork(generator, discriminator, config_gan,
                                       log_tracker=neptune_manager,
                                       allow_disc_switch_off=True)

    gan.fit(train_dataset, config_gan["epochs"], test_dataset)
