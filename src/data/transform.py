import numpy as np


def nctd_transform(x: np.ndarray, n_features: int) -> np.ndarray:
    """
    Transform the input data for NCTD model.
    This function can be customized based on the specific requirements of the NCTD model.
    """

    x = np.tile(x * 255, (n_features, 1))
    x = np.array([np.roll(row, -i) for i, row in enumerate(x)])

    return np.tile(x, (2, 2))
