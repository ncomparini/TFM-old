import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, concatenate, ZeroPadding2D
from tensorflow_core.python.keras.layers import Flatten, Dense


class Discriminator:
    def __init__(self, config, patch_gan, loss_identifier=None):
        self.sample_image_dimensions = config["width"], config["height"], config["sample-image-channels"]
        self.target_image_dimensions = config["width"], config["height"], config["target-image-channels"]
        self.kernel_size = config["kernel-size"]
        self.mean = config["mean"]
        self.std_dev = config["std-dev"]
        self.patch_gan = patch_gan

        loss_identifier = BinaryCrossentropy(from_logits=True) if loss_identifier is None else loss_identifier
        self.loss = tf.keras.losses.get(loss_identifier)

        self.model = self.__create_architecture()

    def __create_architecture(self):
        initializer = tf.random_normal_initializer(self.mean, self.std_dev)

        inp = tf.keras.layers.Input(shape=self.sample_image_dimensions, name='sample_image')
        tar = tf.keras.layers.Input(shape=self.target_image_dimensions, name='target_image')

        con = concatenate([inp, tar])  # (bs, 256, 256, gen_channels + tar_channels)

        down1 = self._downsample(64, 4, self.mean, self.std_dev, apply_batch_norm=False)(con)  # (bs, 128, 128, 64)
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

        if not self.patch_gan:
            flatten = Flatten()(last)
            last = Dense(1)(flatten)

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
