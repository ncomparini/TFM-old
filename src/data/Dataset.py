import tensorflow as tf

from definitions import SEED


class Dataset:
    def __init__(self, batch_size, tag):
        self.batch_size = batch_size
        self.tag = tag
        self.dataset = None

    def create_dataset(self, tensors_list, load_function, buffer_size, shuffle):
        dataset = tf.data.Dataset.from_tensor_slices(tensors_list)
        dataset = dataset.map(load_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=False, seed=SEED)
        dataset = dataset.batch(self.batch_size)

        return dataset
