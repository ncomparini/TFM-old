import enum
from abc import ABC, abstractmethod

from src.data.Dataset import Dataset
import tensorflow as tf
import os
from src.models.gan.discriminator.Discriminator import Discriminator
from src.models.gan.generator.Generator import Generator
from tensorflow.keras.optimizers import Adam
from src.models.log.LogTracker import LogTracker
from datetime import datetime
from definitions import MODELS_PATH


class GanType(enum.Enum):
    IMAGE_GAN = "pix2pix"
    POINT_GAN = "point-gan"


class GenerativeAdversarialNetwork(ABC):
    def __init__(self, generator: Generator, discriminator: Discriminator, config, gan_type: GanType, log_tracker,
                 generator_optimizer=None, discriminator_optimizer=None, allow_disc_switch_off=True):
        self.generator = generator
        self.discriminator = discriminator
        self.params = config
        self.gan_type = gan_type
        self.log_tracker: LogTracker = log_tracker

        self.lr_generator = config["lr-generator"]
        self.lr_discriminator = config["lr-discriminator"]
        self.beta = config["beta"]

        self.generator_optimizer = self._get_optimizer(generator_optimizer, self.lr_generator, self.beta)
        self.discriminator_optimizer = self._get_optimizer(discriminator_optimizer, self.lr_discriminator, self.beta)
        self.allow_disc_switch_off = allow_disc_switch_off

    def fit(self, train_dataset: Dataset, epochs: int, test_dataset: Dataset, samples_to_save=3):
        train_generator = True
        train_discriminator = True

        disc_loss = gen_total_loss = gen_loss = gen_l1_loss = -1

        self._save_params()
        self._set_tags()
        print("Started training...")
        for epoch in range(epochs):
            print(f"epoch: {epoch:03d}/{epochs}", end="\n")

            for n, (sample_image, target_image) in train_dataset.dataset.enumerate():
                disc_loss, gen_total_loss, gen_loss, gen_l1_loss = self._train_step(sample_image,
                                                                                    target_image,
                                                                                    train_generator,
                                                                                    train_discriminator)

            self.save_image_examples(test_dataset, samples_to_save, epoch, epochs)

            self.log_tracker.log_variable("train/generator/gen_total_loss", gen_total_loss)
            self.log_tracker.log_variable("train/generator/gen_loss", gen_loss)
            self.log_tracker.log_variable("train/generator/gen_l1_loss", gen_l1_loss)
            self.log_tracker.log_variable("train/discriminator/disc_loss", disc_loss)

            if self.allow_disc_switch_off:
                self.log_tracker.log_variable("train/discriminator/training-enabled", int(train_discriminator))

                if epoch > 10:
                    if disc_loss < 1:
                        train_discriminator = False
                    else:
                        train_discriminator = True

        print("Finished training")
        self.save_image_examples(train_dataset, 10, epochs, epochs)
        self.save_image_examples(test_dataset, 10, epochs, epochs)

        self.save_model()

    @abstractmethod
    def _train_step(self, sample_image, target_image, generator_training, discriminator_training):
        pass

    def _get_optimizer(self, optimizer_identifier, learning_rate, beta):
        default_optimizer = Adam(learning_rate, beta)
        opt = default_optimizer if optimizer_identifier is None else optimizer_identifier

        return tf.keras.optimizers.get(opt)

    def _save_params(self):
        for key, value in self.params.items():
            self.log_tracker.save_static_variable(f"params/{key}", value)

    def _set_tags(self):
        tags = ["GAN", self.gan_type.value]

        if self.discriminator.patch_gan:
            tags.append("patch-gan")
        else:
            tags.append("vanilla")

        self.log_tracker.add_tags(tags)

    def save_model(self):
        run_id = self.log_tracker.get_run_id()

        print(f"Saving model with run id {run_id}.")
        model_dir = f"model_{run_id}_{datetime.today().strftime('%Y-%m-%d_%H.%M.%S')}"
        model_path = os.path.join(MODELS_PATH, model_dir)
        os.makedirs(model_path)

        with open(os.path.join(model_path, "generator_model.json"), "w") as json_file:
            json_file.write(self.generator.model.to_json())

        with open(os.path.join(model_path, "discriminator_model.json"), "w") as json_file:
            json_file.write(self.discriminator.model.to_json())

        self.generator.model.save(filepath=os.path.join(model_path, "generator_model.h5"))
        self.discriminator.model.save(filepath=os.path.join(model_path, "discriminator_model.h5"))
        self.log_tracker.save_static_variable("model-name", model_dir)
        print("Model saved")

    @abstractmethod
    def save_image_examples(self, dataset: Dataset, num_examples, epoch, total_epochs):
        pass
