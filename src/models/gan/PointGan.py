import numpy as np
import tensorflow as tf

from src.data.Dataset import Dataset
from src.models.gan.GenerativeAdversarialNetwork import GenerativeAdversarialNetwork, GanType
from src.models.gan.discriminator.Discriminator import Discriminator
from src.models.gan.generator.Generator import Generator


class PointGan(GenerativeAdversarialNetwork):
    def __init__(self, generator: Generator, discriminator: Discriminator, config, generator_optimizer=None,
                 discriminator_optimizer=None, log_tracker=None, allow_disc_switch_off=True):
        super().__init__(generator, discriminator, config, GanType.POINT_GAN, log_tracker, generator_optimizer,
                         discriminator_optimizer, allow_disc_switch_off)

    @tf.function
    def _train_step(self, sample_image, target_image, generator_training, discriminator_training):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generator_output = self.generator.model(sample_image, training=generator_training)
            target_image = tf.cast(target_image, dtype=tf.float32)
            disc_real_output = self.discriminator.model([sample_image, target_image], training=discriminator_training)
            generator_output = tf.reshape(generator_output, [-1, 68, 2])
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

    def save_image_examples(self, dataset: Dataset, num_examples, epoch, total_epochs):
        for sample_test_image, target_test_image in dataset.dataset.take(num_examples):
            if epoch == 0 or epoch == total_epochs:
                self.log_tracker.log_image(f"{dataset.tag}/images/sample", sample_test_image[0] * 0.5 + 0.5)

                reshaped_target_points = np.reshape(target_test_image[0], (68, 2))

                target_test_image = np.zeros((128, 128))
                target_test_image[tuple(zip(*reshaped_target_points))] = 1

                self.log_tracker.log_image(f"{dataset.tag}/images/target", target_test_image)

            predicted_points = self.generator.model(sample_test_image)[0]

            reshaped_points = np.reshape(tf.cast(predicted_points, tf.int8), (68, 2))

            output_image = np.zeros((128, 128))
            output_image[tuple(zip(*reshaped_points))] = 1

            self.log_tracker.log_image(f"{dataset.tag}/images/predicted/epoch-{epoch:03d}", output_image)
