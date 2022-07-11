from abc import ABC, abstractmethod
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf


class Discriminator(ABC):
    def __init__(self, config, loss_identifier=None, patch_gan=None):
        self.sample_dimensions = None
        self.target_dimensions = None
        self.kernel_size = config["kernel-size"]
        self.mean = config["mean"]
        self.std_dev = config["std-dev"]
        self.patch_gan = patch_gan

        loss_identifier = BinaryCrossentropy(from_logits=True) if loss_identifier is None else loss_identifier
        self.loss = tf.keras.losses.get(loss_identifier)

        self.model = None

    @abstractmethod
    def create_architecture(self):
        pass

    @abstractmethod
    def compute_loss(self, real_output, generated_output):
        pass
