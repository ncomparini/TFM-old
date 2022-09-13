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

    type_config = config["gan"]["image-to-image"]
    gan_config = type_config[MODEL_TYPE]
    data_config = type_config["data"]

    train_sample_path, train_target_path = get_menpo_paths(data_config, "train", True)
    test_sample_path, test_target_path = get_menpo_paths(data_config, "test", True)

    params_config = gan_config["hyperparameters"]
    batch_size = params_config["batch-size"]

    train_dataset = ImageDataset(train_sample_path, train_target_path, batch_size, shuffle=True, tag="train")
    test_dataset = ImageDataset(test_sample_path, test_target_path, batch_size, shuffle=False, tag="test")

    generator = Image2ImageGenerator(params_config)
    discriminator = ImageDiscriminator(params_config, patch_gan=False)
    neptune_manager = LogTracker(secrets["neptune"])

    gan = ImageGan(generator, discriminator, params_config,
                   log_tracker=neptune_manager,
                   allow_disc_switch_off=False)

    gan.fit(train_dataset, params_config["epochs"], test_dataset)
