import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Dropout, ReLU, LeakyReLU, Dense, \
    Flatten

from src.models.gan.generator.Generator import Generator


class Image2PointGenerator(Generator):
    def __init__(self, config, loss_identifier=None, pix2pix=False):
        super().__init__(config, loss_identifier, pix2pix)

        self.input_dimensions = config["width"], config["height"], config["sample-image-channels"]
        self.output_dimensions = (config["number-of-landmarks"], 2)

        self.model = self.create_architecture()

    def create_architecture(self):
        inputs = tf.keras.layers.Input(shape=self.input_dimensions)

        down_stack = [
            self._downsample(64, self.kernel_size, self.mean, self.std_dev, apply_batchnorm=False),
            # (bs, 128, 128, 64)
            self._downsample(128, self.kernel_size, self.mean, self.std_dev),  # (bs, 64, 64, 128)
            self._downsample(256, self.kernel_size, self.mean, self.std_dev),  # (bs, 32, 32, 256)
            self._downsample(512, self.kernel_size, self.mean, self.std_dev),  # (bs, 16, 16, 512)
            self._downsample(512, self.kernel_size, self.mean, self.std_dev),  # (bs, 8, 8, 512)
            self._downsample(512, self.kernel_size, self.mean, self.std_dev),  # (bs, 4, 4, 512)
            self._downsample(512, self.kernel_size, self.mean, self.std_dev),  # (bs, 2, 2, 512)
            # self.downsample(512, self.kernel_size),                         # (bs, 1, 1, 512)
        ]

        initializer = tf.random_normal_initializer(self.mean, self.std_dev)
        last_conv = Conv2DTranspose(filters=1,
                                    kernel_size=self.kernel_size,
                                    strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    activation=tf.keras.activations.relu)
        x = inputs
        for down in down_stack:
            x = down(x)

        x = last_conv(x)

        flatten_last = Flatten()(x)
        num_final_units = self.output_dimensions[0] * self.output_dimensions[1]
        last = Dense(units=num_final_units)(flatten_last)

        return tf.keras.Model(inputs=inputs, outputs=last, name="Generator")

    def _upsample(self, filters, size, mean, std_dev, prob_dropout=None):
        initializer = tf.random_normal_initializer(mean, std_dev)

        block = tf.keras.Sequential()
        block.add(Conv2DTranspose(filters,
                                  kernel_size=size,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  use_bias=False))

        block.add(BatchNormalization())

        if prob_dropout:
            block.add(Dropout(prob_dropout))

        block.add(ReLU())

        return block

    def _downsample(self, filters, size, mean, std_dev, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(mean, std_dev)

        block = tf.keras.Sequential()
        block.add(Conv2D(filters,
                         kernel_size=size,
                         strides=2,
                         padding='same',
                         kernel_initializer=initializer,
                         use_bias=not apply_batchnorm))

        if apply_batchnorm:
            block.add(BatchNormalization())

        block.add(LeakyReLU())

        return block

    def compute_loss(self, disc_generated_output, gen_output, target):
        generator_loss = self.loss(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_generator_loss = generator_loss + (self.lambda_ * l1_loss)

        return total_generator_loss, generator_loss, l1_loss
