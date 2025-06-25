import time
from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchinfo import summary

from data.cdc import IgtdDataModule, NctdDataModule
from data.transform import nctd_transform
from model.classifier import DiabetesRiskClassifier
from model.cnn import ConvNet
from model.model_type import ModelType
from train_and_eval.evaluate import compute_metrics
from train_and_eval.metrics_logger import MetricsLogger


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    torch.manual_seed(seed=42)

    # Init data module
    if cfg.model.name == ModelType.NCTD.value:
        dm = NctdDataModule(
            data_file=Path(cfg.data.file_path),
            target_col=cfg.data.target_col,
        )
        dm.setup(transform=nctd_transform, downsample=cfg.data.downsample)

    elif cfg.model.name == ModelType.IGTD.value:
        dm = IgtdDataModule(
            data_file=Path(cfg.data.file_path),
            img_dir=Path(cfg.igtd.img_dir),
            target_col=cfg.data.target_col,
        )
        dm.setup()

    else:
        raise ValueError(f"Unsupported model type: {cfg.model.name}")

    # Init CNN model
    logger.info("Initializing CNN model...")
    model = ConvNet()
    summary(
        model, input_size=(1, 1, cfg.model.input_dim.nrows, cfg.model.input_dim.ncols)
    )

    logger.info("Training the model...")
    metrics_logger = MetricsLogger()
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=5,
        verbose=True,
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # same metric as EarlyStopping
        mode="min",
        save_top_k=1,  # keep only the best model
        filename="best-checkpoint",
        verbose=False,
        save_weights_only=False,  # save full model (or True for just weights)
    )
    classifier = DiabetesRiskClassifier(
        model=model, batch_size=cfg.train.batch_size, pos_weight=cfg.model.pos_weight
    )
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        devices=cfg.train.devices,
        enable_progress_bar=True,
        log_every_n_steps=cfg.train.log_every_n_steps,
        enable_model_summary=False,  # optional
        callbacks=[metrics_logger, early_stop_callback, checkpoint_callback],
    )

    start_time = time.time()
    trainer.fit(
        classifier,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )
    end_time = time.time()

    logger.info(f"Training time: {(end_time - start_time):.2f} seconds")

    # Load the best model and test the performance
    classifier = DiabetesRiskClassifier.load_from_checkpoint(
        checkpoint_callback.best_model_path,
    )
    output = trainer.test(classifier, dm.test_dataloader())
    logger.debug(f"Test output: {output}")

    model.eval()
    y_pred = []
    y_test = []

    with torch.no_grad():
        for x, y in dm.test_dataloader():

            logits = model(x)  # [B, 1]
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().squeeze()

            y_pred.extend(preds.cpu().numpy())
            y_test.extend(y.cpu().numpy())

    results = [compute_metrics(y_test, y_pred, avg) for avg in cfg.results.avg_options]
    results_df = pd.DataFrame(results)
    results_df.to_csv(cfg.results.file_path, index=False)


if __name__ == "__main__":
    main()
