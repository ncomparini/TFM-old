import os

from constants import *
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD


class FECModel:

    def __init__(self, lr=0.001, model=None, optimizer=Adam(learning_rate=0.001)):
        self.model = model
        self.learning_rate = lr
        self.optimizer = optimizer

        self.checkpoint = None
        self.checkpoint_prefix = None
        self.checkpoint_manager = None


    def build_model(self):
        model = tf.keras.Sequential()

        model.add(Conv2D(filters=32,
                         kernel_size=(3, 3),
                         strides=1,
                         activation=tf.nn.relu,
                         input_shape=(HEIGHT, WIDTH, CHANNELS)))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(BatchNormalization())
        model.add(Dropout(DROPOUT))
        model.add(Conv2D(filters=64,
                         kernel_size=(3, 3),
                         strides=1,
                         activation=tf.nn.relu))
        model.add(BatchNormalization())
        model.add(Dropout(DROPOUT * 2))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(units=512,
                        activation=tf.nn.relu))
        model.add(Dense(units=N_OUTPUT_LABELS,
                        activation=tf.nn.softmax))

        self.model = model

        return self

    def set_optimizer(self, name="SGD"):
        if name == "SGD":
            self.optimizer = SGD(lr=self.learning_rate)
        elif name == "Adam":
            self.optimizer = Adam(learning_rate=self.learning_rate)
        else:
            raise Exception("Error. Choose a valid optimizer: \"SGD\" or \"Adam\"")

        return self

    def compile_model(self, loss_function, metrics):
        if not self.optimizer:
            raise Exception("Error. You must first set an optimizer.")

        self.model.compile(optimizer=self.optimizer,
                           loss=loss_function,
                            metrics=metrics)

        return self

    # TODO: Set checkpoints with neptune.ai

    def initialize_checkpoints(self):
        self.checkpoint_prefix = os.path.join(PATH_CHECKPOINTS, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, PATH_CHECKPOINTS, max_to_keep=2)

        return self

    def fit_model(self, x_train, y_train, x_test, y_test, force_training=False):
        if not x_train or not y_train:
            raise Exception("Error. Training slices must be defined")
        if not x_test or not y_test:
            raise Exception("Error. Testing slices must be defined")

        if not force_training and self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        else:
            self.model.fit(x_train, y_train,
                           epochs=EPOCHS,
                           validation_data=(x_test, y_test),
                           batch_size=BATCH_SIZE,
                           verbose=2)

        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

        return self
