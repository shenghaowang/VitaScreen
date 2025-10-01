from typing import Tuple

import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from model.ensemble_tree import EnsembleTreeClassifier, Pool
from train_and_eval.base_trainer import BaseTrainer
from train_and_eval.evaluate import compute_metrics


class EnsembleTreeTrainer(BaseTrainer):
    """
    Trainer class for ensemble tree-based models, in particular
    CatBoostClassifier on CDC data.
    """

    def __init__(self, hyperparams: DictConfig):
        super().__init__()
        self.hyperparams = OmegaConf.to_container(hyperparams, resolve=True)
        self.best_model = None

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
        best_cv_metrics = None

        for i, (train_idx, val_idx) in enumerate(self.k_fold_indices):
            logger.info(f"Training fold {i + 1}/{len(self.k_fold_indices)}")

            train_pool = Pool(data=self.X[train_idx], label=self.y[train_idx])
            val_pool = Pool(data=self.X[val_idx], label=self.y[val_idx])

            model = EnsembleTreeClassifier(**self.hyperparams)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

            y_preds = model.predict(self.X[val_idx])
            val_metrics = compute_metrics(self.y[val_idx], y_preds, avg_option="macro")

            logger.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Validation Precision: {val_metrics['precision']:.4f}")
            logger.info(f"Validation Recall: {val_metrics['recall']:.4f}")
            logger.info(f"Validation F1 Score: {val_metrics['f1_score']:.4f}")

            if val_metrics["f1_score"] > best_f1_score:
                best_f1_score = val_metrics["f1_score"]
                self.best_model = model
                best_cv_metrics = val_metrics

        logger.info("Test validation metrics of the best model:")
        for metric, value in best_cv_metrics.items():
            logger.info(f"Best Model {metric}: {value}")

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
