from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data.utils import split_data
from model.ensemble_tree import EnsembleTreeClassifier, Pool
from train_and_eval.evaluate import compute_metrics


class EnsembleTreeTrainer:
    """
    Trainer class for ensemble tree-based models, in particular
    CatBoostClassifier on CDC data.
    """

    def __init__(self, hyperparams: DictConfig):
        self.hyperparams = OmegaConf.to_container(hyperparams, resolve=True)
        self.best_model = None

    def setup(self, data_cfg: DictConfig):
        """
        Load and prepare the CDC dataset for training, validation, and testing.

        Parameters
        ----------
        data_cfg : DictConfig
            Hydra/OmegaConf configuration containing:
                - file_path: Path to the CDC dataset file.
                - target_col: Name of the target variable column.
                - downsample: Whether to downsample the data for addressing class imbalance.
        """
        df = pd.read_csv(data_cfg.file_path)

        feature_cols = [col for col in df.columns if col != data_cfg.target_col]
        self.X, self.y = df[feature_cols].values, df[data_cfg.target_col].values

        k_fold_indices, test_idx = split_data(
            X=self.X,
            y=self.y,
            downsample=data_cfg.downsample,
        )

        self.k_fold_indices = k_fold_indices
        self.test_idx = test_idx

    def train(self) -> Tuple[np.ndarray, np.ndarray]:
        """Train the ensemble tree model."""
        # Pick the first fold from k-fold indices
        train_idx, val_idx = self.k_fold_indices[0]
        logger.info(
            f"Size of training set: {len(train_idx)}, validation set: {len(val_idx)}"
        )

        train_pool = Pool(data=self.X[train_idx], label=self.y[train_idx])
        val_pool = Pool(data=self.X[val_idx], label=self.y[val_idx])

        model = EnsembleTreeClassifier(**self.hyperparams)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

        y_preds = model.predict(self.X[val_idx])

        return self.y[val_idx], y_preds

    def cross_validate(self):
        """Train the ensemble tree model with cross validation."""
        best_f1_score = 0.0

        for i, (train_idx, val_idx) in enumerate(self.k_fold_indices):
            logger.info(f"Training fold {i + 1}/{len(self.k_fold_indices)}")

            train_pool = Pool(data=self.X[train_idx], label=self.y[train_idx])
            val_pool = Pool(data=self.X[val_idx], label=self.y[val_idx])

            model = EnsembleTreeClassifier(**self.hyperparams)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

            y_preds = model.predict(self.X[val_idx])
            val_metrics = compute_metrics(self.y[val_idx], y_preds, avg_option="binary")

            logger.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Validation Precision: {val_metrics['precision']:.4f}")
            logger.info(f"Validation Recall: {val_metrics['recall']:.4f}")
            logger.info(f"Validation F1 Score: {val_metrics['f1_score']:.4f}")

            if val_metrics["f1_score"] > best_f1_score:
                best_f1_score = val_metrics["f1_score"]
                self.best_model = model

    def evaluate(self):
        """
        Evaluate the ensemble tree model on validation and test sets.

        Returns
        -------
        y_test : np.ndarray
            True labels for the test set.
        y_preds : np.ndarray
            Predicted labels for the test set.
        """
        y_pred = self.best_model.predict(self.X[self.test_idx])

        return self.y[self.test_idx], y_pred
