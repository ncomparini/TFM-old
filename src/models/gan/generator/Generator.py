from abc import ABC, abstractmethod
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf


class Generator(ABC):
    def __init__(self, config, loss_identifier=None, pix2pix=None):
        self.input_dimensions = None
        self.output_dimensions = None
        self.kernel_size = config["kernel-size"]
        self.mean = config["mean"]
        self.std_dev = config["std-dev"]
        self.prob_dropout = config["dropout"]
        self.lambda_ = config["lambda"]
        self.pix2pix = pix2pix

        loss_identifier = BinaryCrossentropy(from_logits=True) if loss_identifier is None else loss_identifier
        self.loss = tf.keras.losses.get(loss_identifier)

        self.model = None

    @abstractmethod
    def create_architecture(self):
        pass

    @abstractmethod
    def compute_loss(self, disc_generated_output, gen_output, target):
        pass
