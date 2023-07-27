import tensorflow as tf
import pandas as pd


def input_fun(features, labels, training=True, batch_size=256):
    # Convert the input to a dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # shuffle and repeat if you are training mode
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)
