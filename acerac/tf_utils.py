from typing import Union, Tuple

import numpy as np
import tensorflow as tf


def kronecker_prod(x: Union[tf.Tensor, np.array], y: Union[tf.Tensor, np.array]) -> tf.Tensor:
    """Computes Kronecker product between x and y.

    Args:
        x: First tensor.
        y: Second tensor.

    Returns:
        Product of x and y.
    """
    operator_1 = tf.linalg.LinearOperatorFullMatrix(x)
    operator_2 = tf.linalg.LinearOperatorFullMatrix(y)
    prod = tf.linalg.LinearOperatorKronecker([operator_1, operator_2]).to_dense()
    return prod
