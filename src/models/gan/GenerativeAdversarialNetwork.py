from src.data.ImageDataset import ImageDataset
import tensorflow as tf
import os
from src.models.gan.Discriminator import Discriminator
from src.models.gan.Generator import Generator
from tensorflow.keras.optimizers import Adam
from src.models.log.LogTracker import LogTracker
from datetime import datetime
from definitions import MODELS_PATH


class GenerativeAdversarialNetwork:
    def __init__(self, generator: Generator, discriminator: Discriminator, config,
                 generator_optimizer=None, discriminator_optimizer=None, log_tracker=None, allow_disc_switch_off=True):
        self.generator = generator
        self.discriminator = discriminator
        self.params = config
        self.lr_generator = config["lr-generator"]
        self.lr_discriminator = config["lr-discriminator"]
        self.beta = config["beta"]

        self.generator_optimizer = self._get_optimizer(generator_optimizer, self.lr_generator, self.beta)
        self.discriminator_optimizer = self._get_optimizer(discriminator_optimizer, self.lr_discriminator, self.beta)
        self.log_tracker: LogTracker = log_tracker
        self.allow_disc_switch_off = allow_disc_switch_off

    def fit(self, train_dataset: ImageDataset, epochs: int, test_dataset: ImageDataset, samples_to_save=3):
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

                if epoch > 4:
                    if disc_loss < 1:
                        train_discriminator = False
                    else:
                        train_discriminator = True

        print("Finished training")
        self.save_image_examples(train_dataset, 10, epochs, epochs)
        self.save_image_examples(test_dataset, 10, epochs, epochs)

        self.save_model()

    @tf.function
    def _train_step(self, sample_image, target_image, generator_training, discriminator_training):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generator_output = self.generator.model(sample_image, training=generator_training)

            disc_real_output = self.discriminator.model([sample_image, target_image], training=discriminator_training)
            disc_generated_output = self.discriminator.model([sample_image, generator_output],
                                                             training=discriminator_training)

            gen_total_loss, gen_loss, gen_l1_loss = self.generator.compute_loss(disc_generated_output,
                                                                                generator_output,
                                                                                target_image)
            disc_loss = self.discriminator.compute_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.model.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.discriminator.model.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                     self.generator.model.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                         self.discriminator.model.trainable_variables))

        return disc_loss, gen_total_loss, gen_loss, gen_l1_loss

    def _get_optimizer(self, optimizer_identifier, learning_rate, beta):
        default_optimizer = Adam(learning_rate, beta)
        opt = default_optimizer if optimizer_identifier is None else optimizer_identifier

        return tf.keras.optimizers.get(opt)

    def _save_params(self):
        for key, value in self.params.items():
            self.log_tracker.save_static_variable(f"params/{key}", value)

    def _set_tags(self):
        tags = ["GAN"]

        if self.generator.pix2pix:
            tags.append("pix2pix")

        if self.discriminator.patch_gan:
            tags.append("patch-gan")
        else:
            tags.append("vanilla")

        self.log_tracker.add_tags(tags)

    def save_model(self):
        print("Saving model...")
        model_path = os.path.join(MODELS_PATH, f"model_{datetime.today().strftime('%Y-%m-%d_%H.%M.%S')}")
        os.makedirs(model_path)

        with open(os.path.join(model_path, "generator_model.json"), "w") as json_file:
            json_file.write(self.generator.model.to_json())

        with open(os.path.join(model_path, "discriminator_model.json"), "w") as json_file:
            json_file.write(self.discriminator.model.to_json())

        self.generator.model.save_model(filepath=os.path.join(model_path, "generator_weights.h5"))
        self.discriminator.model.save_model(filepath=os.path.join(model_path, "discriminator_weights.h5"))
        self.log_tracker.save_static_variable("model-path", model_path)
        print("Model saved")

    def save_image_examples(self, dataset: ImageDataset, num_examples, epoch, total_epochs):
        for sample_test_image, target_test_image in dataset.dataset.take(num_examples):
            if epoch == 0 or epoch == total_epochs:
                self.log_tracker.log_image(f"{dataset.tag}/images/sample", sample_test_image[0] * 0.5 + 0.5)
                self.log_tracker.log_image(f"{dataset.tag}/images/target", target_test_image[0][..., -1] * 0.5 + 0.5)

            output_image = self.generator.model(sample_test_image)[0][..., -1] * 0.5 + 0.5
            self.log_tracker.log_image(f"{dataset.tag}/images/predicted/epoch-{epoch:03d}", output_image)
