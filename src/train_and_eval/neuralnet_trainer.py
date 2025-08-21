import time
from typing import List, Tuple, Union

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger
from omegaconf import DictConfig
from torchinfo import summary

from data.cdc import IgtdDataModule, NeuralNetDataModule
from model.classifier import DiabetesRiskClassifier
from model.cnn import ConvNet
from model.mlp import MLP
from model.model_type import ModelType
from train_and_eval.metrics_logger import MetricsLogger


class NeuralNetTrainer:
    def __init__(
        self,
        data_module: Union[IgtdDataModule, NeuralNetDataModule],
        model_cfg: DictConfig,
        train_cfg: DictConfig,
    ):
        self.train_loader = data_module.train_dataloader()
        self.val_loader = data_module.val_dataloader()
        self.test_loader = data_module.test_dataloader()

        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

    def init_trainer(self, model: ConvNet | MLP = ConvNet()):
        """Initialize the PyTorch Lightning trainer."""

        self.model = model
        input_size = (
            (1, self.model_cfg.input_dim)
            if self.model_cfg.name == ModelType.MLP.value
            else (1, 1, self.model_cfg.input_dim.nrows, self.model_cfg.input_dim.ncols)
        )
        summary(model=self.model, input_size=input_size)

        metrics_logger = MetricsLogger()
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=5,
            verbose=True,
            mode="min",
        )
        self.checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",  # same metric as EarlyStopping
            mode="min",
            save_top_k=1,  # keep only the best model
            filename="best-checkpoint",
            verbose=False,
            save_weights_only=False,  # save full model (or True for just weights)
        )

        self.trainer = L.Trainer(
            max_epochs=self.train_cfg.max_epochs,
            devices=self.train_cfg.devices,
            enable_progress_bar=True,
            log_every_n_steps=self.train_cfg.log_every_n_steps,
            enable_model_summary=False,  # optional
            callbacks=[metrics_logger, early_stop_callback, self.checkpoint_callback],
        )

    def train(self):
        logger.info("Training the model...")

        classifier = DiabetesRiskClassifier(
            model=self.model,
            batch_size=self.train_cfg.batch_size,
            pos_weight=self.model_cfg.pos_weight,
        )

        start_time = time.time()
        self.trainer.fit(
            classifier,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
        )
        end_time = time.time()

        logger.info(f"Training time: {(end_time - start_time):.2f} seconds")

    def test(self) -> None:
        """Load the best model and test the performance."""

        classifier = DiabetesRiskClassifier.load_from_checkpoint(
            self.checkpoint_callback.best_model_path,
        )
        output = self.trainer.test(classifier, self.test_loader)
        logger.debug(f"Test output: {output}")

    def evaluate(self) -> Tuple[List[float], List[float]]:
        """
        Evaluate the model on the test set.
        """
        self.model.eval()
        y_pred = []
        y_test = []

        with torch.no_grad():
            for x, y in self.test_loader:

                logits = self.model(x)  # [B, 1]
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long().squeeze()

                y_pred.extend(preds.cpu().numpy())
                y_test.extend(y.cpu().numpy())

        return y_test, y_pred
