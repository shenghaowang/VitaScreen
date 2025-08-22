from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours
from loguru import logger
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    X: np.ndarray, y: np.ndarray, downsample: bool = False, n_splits=5
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    indices = np.arange(len(X))

    train_val_idx, test_idx = train_test_split(
        indices, stratify=y, test_size=0.2, random_state=42
    )

    if downsample:
        sampler = EditedNearestNeighbours()
        _, _ = sampler.fit_resample(X[train_val_idx], y[train_val_idx])
        train_val_idx = sampler.sample_indices_

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    return list(cv.split(X[train_val_idx], y[train_val_idx])), test_idx


def prepare_data(
    data_file: Path, target_col: str, downsample: bool = False
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Load and prepare data for training

    Parameters
    ----------
    data_file : Path
        Path to the CSV file containing the dataset
    target_col : str
        Name of the target column in the dataset
    downsample : bool, optional
        Whether to downsample the negative class data, by default False

    Returns
    -------
    Tuple[ Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], ]
        Returns training, validation, and test datasets as tuples of features and labels
    """

    # Load raw data
    df = pd.read_csv(data_file)
    feature_cols = [col for col in df.columns if col != target_col]

    # Split data for training, validation, and testing
    X, y = df[feature_cols].values, df[target_col].values
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if downsample:
        # Resampling using Edited Nearest Neighbours
        enn = EditedNearestNeighbours()
        X_train_val, y_train_val = enn.fit_resample(X_train_val, y_train_val)
        logger.info(
            f"Resampled training data shape: {X_train_val.shape}, {y_train_val.shape}"
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def nctd_transform(x: np.ndarray, n_features: int) -> np.ndarray:
    """
    Transform the input data for NCTD model.
    This function can be customized based on the specific requirements of the NCTD model.
    """

    x = np.tile(x * 255, (n_features, 1))
    x = np.array([np.roll(row, -i) for i, row in enumerate(x)])

    return np.tile(x, (2, 2))
