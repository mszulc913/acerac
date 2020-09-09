from typing import List, Callable

import tensorflow as tf

from utils import normc_initializer


def build_cnn_network(
    filters: tuple = (32, 64, 64), kernels: tuple = (8, 4, 3), strides: tuple = ((4, 4), (2, 2), (1, 1)),
    activation: str = 'relu', initializer: Callable = normc_initializer
) -> List[tf.keras.Model]:
    """Builds convolutional neural network.

    Args:
        filters: Tuple with filters to be used.
        kernels: Tuple with kernel sizes to be used.
        strides: Tuple with strides to be used
        activation: Activation function to be used.
        initializer: Callable to weights initializer function.

    Returns:
        Created network (list of Keras layers).
    """
    assert len(filters) == len(kernels) == len(strides), \
        f"Layers' specifications must have the same lengths. " \
        f"Got: len(filters)=={len(filters)}, len(kernels)=={len(kernels)}, len(strides)=={len(strides)}"

    expand_layer = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))
    cast_layer = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))
    layers = [expand_layer, cast_layer] + [
        tf.keras.layers.Conv2D(
            cnn_filter,
            kernel,
            strides=stride,
            activation=activation,
            kernel_initializer=initializer(),
            padding="same"
        ) for cnn_filter, kernel, stride in zip(filters, kernels, strides)
    ]

    layers.append(tf.keras.layers.Flatten())
    return layers
