import tensorflow as tf
import pandas as pd


def get_model(feature_columns):
    # Build a DNN with 3 hidden layers with 30, 20 and 10 nodes each
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        # three hidden layers of 30, 20 and 10 nodes respectively
        hidden_units=[30, 20, 10],
        # The model must choose between 3 classes
        n_classes=3
    )
    return classifier
