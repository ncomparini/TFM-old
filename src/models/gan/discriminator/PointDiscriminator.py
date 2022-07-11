import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Concatenate, ZeroPadding2D
from tensorflow_core.python.keras.layers import Flatten, Dense

from src.models.gan.discriminator.Discriminator import Discriminator


class PointDiscriminator(Discriminator):
    def __init__(self, config, loss_identifier=None, patch_gan=None):
        super().__init__(config, loss_identifier, patch_gan)
        self.sample_dimensions = config["width"], config["height"], config["sample-image-channels"]
        self.target_dimensions = (config["number-of-landmarks"], 2)

        self.model = self.create_architecture()

    def create_architecture(self):
        initializer = tf.random_normal_initializer(self.mean, self.std_dev)

        inp = tf.keras.layers.Input(shape=self.sample_dimensions, name='sample_image')
        tar = tf.keras.layers.Input(shape=self.target_dimensions, name='target_points')

        down1 = self._downsample(64, 4, self.mean, self.std_dev, apply_batch_norm=False)(inp)  # (bs, 128, 128, 64)
        down2 = self._downsample(128, 4, self.mean, self.std_dev)(down1)  # (bs, 64, 64, 128)
        down3 = self._downsample(256, 4, self.mean, self.std_dev)(down2)  # (bs, 32, 32, 256)

        zero_pad1 = ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = Conv2D(filters=512,
                      kernel_size=self.kernel_size,
                      strides=1,
                      kernel_initializer=initializer,
                      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

        batch_norm = BatchNormalization()(conv)
        leaky_relu = LeakyReLU()(batch_norm)
        zero_pad2 = ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

        last = Conv2D(filters=1,
                      kernel_size=self.kernel_size,
                      strides=1,
                      kernel_initializer=initializer,
                      activation=tf.keras.activations.linear)(zero_pad2)  # (bs, 30, 30, 1)

        flatten_last = Flatten()(last)
        flatten_points = tf.cast(Flatten()(tar), dtype=tf.float32)

        concatenated = Concatenate()([flatten_last, flatten_points])

        last = Dense(1)(concatenated)

        return tf.keras.Model(inputs=[inp, tar], outputs=last, name="Discriminator")

    def _downsample(self, filters, size, mean, std_dev, apply_batch_norm=True):
        initializer = tf.random_normal_initializer(mean, std_dev)

        block = tf.keras.Sequential()
        block.add(Conv2D(filters,
                         kernel_size=size,
                         strides=2,
                         padding='same',
                         kernel_initializer=initializer,
                         use_bias=not apply_batch_norm))

        if apply_batch_norm:
            block.add(BatchNormalization())

        block.add(LeakyReLU())

        return block

    def compute_loss(self, real_output, generated_output):
        real_loss = self.loss(tf.ones_like(real_output), real_output)
        generated_loss = self.loss(tf.zeros_like(generated_output), generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss
