import importlib
from enum import Enum
from typing import List, Tuple

import numpy as np
from loguru import logger
from sklearn.model_selection import StratifiedKFold, train_test_split


class ClassImbalanceHandler(Enum):
    smote = "imblearn.over_sampling.SMOTE"
    rus = "imblearn.under_sampling.RandomUnderSampler"
    enn = "imblearn.under_sampling.EditedNearestNeighbours"

    @classmethod
    def get_handler(cls, name: str):
        """Get handler by algorithm name (e.g., 'smote', 'enn', 'rus')"""
        try:
            handler = cls[name.lower()]
            module_name, method_name = handler.value.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, method_name)

        except KeyError:
            available = ", ".join([h.name for h in cls])
            raise ValueError(
                f"Unknown handler '{name}'. Available handlers: {available}"
            )

        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import {method_name} from {module_name}: {e}")


def split_data(
    X: np.ndarray, y: np.ndarray, resampler_name: str = None, n_splits=5
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    indices = np.arange(len(X))

    train_val_idx, test_idx = train_test_split(
        indices, stratify=y, test_size=0.2, random_state=42
    )

    if resampler_name is not None:
        Resampler = ClassImbalanceHandler.get_handler(resampler_name)

        # Try with random_state first, fallback without it
        try:
            resampler = Resampler(random_state=42)

        except TypeError:
            resampler = Resampler()

        # Check if this is an undersampling method with sample_indices_
        if hasattr(resampler, "sample_indices_"):
            # For undersampling methods like EditedNearestNeighbours
            resampler.fit_resample(X[train_val_idx], y[train_val_idx])

            # Map to global indices
            train_val_idx = train_val_idx[resampler.sample_indices_]

        else:
            # For oversampling methods like SMOTE that don't have sample_indices_
            # We'll apply resampling during cross-validation, not here
            # Just log that resampling will be applied later
            logger.info(
                f"Resampling with {resampler_name} will be applied during cross-validation"
            )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []

    # Indices returned by cv are LOCAL to train_val_idx; map them to GLOBAL
    for tr_local, va_local in cv.split(np.zeros(len(train_val_idx)), y[train_val_idx]):
        tr_global = train_val_idx[tr_local]
        va_global = train_val_idx[va_local]
        folds.append((tr_global, va_global))

    return folds, test_idx


def nctd_transform(x: np.ndarray, n_features: int) -> np.ndarray:
    """
    Transform the input data for NCTD model.
    This function can be customized based on the specific requirements of the NCTD model.
    """

    x = np.tile(x * 255, (n_features, 1))
    x = np.array([np.roll(row, -i) for i, row in enumerate(x)])

    return np.tile(x, (2, 2))
