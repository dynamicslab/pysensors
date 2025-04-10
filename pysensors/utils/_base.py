"""
Various utility functions.
"""

import numpy as np


def validate_input(x, sensors=None):
    """
    Ensure that x is of compatible type and shape.

    Parameters
    ----------

    x: numpy ndarray, shape [n_features,] or [n_examples, n_features]
        Data to be validated.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("x must be a numpy array")

    if sensors is not None:
        n_features = len(x) if np.ndim(x) == 1 else x.shape[1]
        if len(sensors) != n_features:
            raise ValueError(
                """x has the wrong number of features: {}.
                Expected {}""".format(
                    n_features, len(sensors)
                )
            )

    return x
