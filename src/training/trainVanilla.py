from src.models.gan.ImageGan import ImageGan
from src.models.gan.generator.Image2ImageGenerator import Image2ImageGenerator
from src.models.gan.discriminator.ImageDiscriminator import ImageDiscriminator
from src.models.log.LogTracker import LogTracker
from src.data.ImageDataset import ImageDataset
from definitions import CONFIG_PATH, SECRETS_PATH, get_menpo_paths
import yaml

MODEL_TYPE = "vanilla"

if __name__ == "__main__":
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)

    with open(SECRETS_PATH, 'r') as file:
        secrets = yaml.safe_load(file)

    train_sample_path, train_target_path = get_menpo_paths("train", True)
    test_sample_path, test_target_path = get_menpo_paths("test", True)

    config_gan = config["hyperparameters"]["gan"][MODEL_TYPE]
    batch_size = config_gan["batch-size"]

    train_dataset = ImageDataset(train_sample_path, train_target_path, batch_size, shuffle=True, tag="train")
    test_dataset = ImageDataset(test_sample_path, test_target_path, batch_size, shuffle=False, tag="test")

    generator = Image2ImageGenerator(config_gan)
    discriminator = ImageDiscriminator(config_gan, patch_gan=False)
    neptune_manager = LogTracker(secrets["neptune"])

    gan = ImageGan(generator, discriminator, config_gan,
                   log_tracker=neptune_manager,
                   allow_disc_switch_off=False)

    gan.fit(train_dataset, config_gan["epochs"], test_dataset)
