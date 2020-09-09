from typing import List, Callable, Tuple

import tensorflow as tf

from utils import normc_initializer


def build_mlp_network(
    layers_sizes: Tuple[int] = (256, 256), activation: str = 'tanh', initializer: Callable = normc_initializer
) -> List[tf.keras.Model]:
    """Builds feedforward neural network.

    Args:
        layers_sizes: List of sizes of hidden layers.
        activation: Activation function name.
        initializer: Callable to weights initializer function.

    Returns:
        Created network (list of Keras layers)
    """
    layers = [
        tf.keras.layers.Dense(
            layer_size,
            activation=activation,
            kernel_initializer=initializer()
        ) for layer_size in layers_sizes
    ]

    return layers
