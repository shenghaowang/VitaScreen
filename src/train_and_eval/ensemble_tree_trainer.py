from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data.cdc import prepare_data
from model.ensemble_tree import EnsembleTreeClassifier, Pool
from train_and_eval.evaluate import compute_metrics


class EnsembleTreeTrainer:
    def __init__(self, hyperparams: DictConfig):
        hyperparams = OmegaConf.to_container(hyperparams, resolve=True)
        self.model = EnsembleTreeClassifier(**hyperparams)

    def setup(self, data_cfg: DictConfig):
        """Load and prepare the dataset."""
        (
            (X_train, y_train),
            (self.X_val, self.y_val),
            (self.X_test, self.y_test),
        ) = prepare_data(
            data_file=data_cfg.file_path,
            target_col=data_cfg.target_col,
            downsample=data_cfg.downsample,
        )

        self.train_pool = Pool(data=X_train, label=y_train)
        self.val_pool = Pool(data=self.X_val, label=self.y_val)

    def train(self):
        """Train the model."""
        self.model.fit(
            self.train_pool, eval_set=self.val_pool, early_stopping_rounds=50
        )

    def evaluate(self):
        """Evaluate the model."""
        y_preds = self.model.predict(self.X_val)
        val_metrics = compute_metrics(self.y_val, y_preds, avg_option="binary")

        logger.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"Validation Precision: {val_metrics['precision']:.4f}")
        logger.info(f"Validation Recall: {val_metrics['recall']:.4f}")
        logger.info(f"Validation F1 Score: {val_metrics['f1_score']:.4f}")

        y_preds = self.model.predict(self.X_test)

        return self.y_test, y_preds
