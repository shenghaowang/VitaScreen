import random
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data.cdc import IgtdDataModule, NeuralNetDataModule
from data.utils import nctd_transform
from model.model_type import ModelType
from train_and_eval.ensemble_tree_trainer import EnsembleTreeTrainer
from train_and_eval.evaluate import compute_metrics
from train_and_eval.neuralnet_trainer import NeuralNetTrainer


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # Enable reproducibility
    torch.manual_seed(seed=42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Data splitting is now handled within each trainer's setup() method

    # Initialize trainer based on model type - all use cross-validation
    transform = nctd_transform if cfg.model.name == ModelType.NCTD.value else None
    match cfg.model.name:
        case ModelType.MLP.value | ModelType.NCTD.value:
            trainer = NeuralNetTrainer(
                model_cfg=cfg.model,
                train_cfg=cfg.train,
            )
            trainer.setup(data_cfg=cfg.data)
            trainer.cross_validate(
                data_file=Path(cfg.data.file_path),
                transform=transform,
                feature_cols=cfg.data.feature_cols
                if "feature_cols" in cfg.data
                else None,
            )

            logger.info("Evaluating the model on the test set ...")
            dm = NeuralNetDataModule(
                data_file=Path(cfg.data.file_path),
                feature_cols=cfg.data.feature_cols
                if "feature_cols" in cfg.data
                else None,
            )
            train_idx, val_idx = trainer.k_fold_indices[0]
            dm.setup(
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=trainer.test_idx,
                transform=transform,
            )
            y_test, y_pred = trainer.evaluate(dm.test_dataloader())

            # Export predicted probabilities
            trainer.export_prob(
                data_file=Path(cfg.data.file_path),
                output_path=Path(cfg.results.prob_path),
                transform=transform,
                feature_cols=cfg.data.feature_cols
                if "feature_cols" in cfg.data
                else None,
            )

        case ModelType.IGTD.value:
            trainer = NeuralNetTrainer(
                model_cfg=cfg.model,
                train_cfg=cfg.train,
            )
            trainer.setup(data_cfg=cfg.data)
            trainer.cross_validate(
                data_file=Path(cfg.data.file_path), img_dir=Path(cfg.igtd.img_dir)
            )

            logger.info("Evaluating the model on the test set ...")
            dm = IgtdDataModule(
                data_file=Path(cfg.data.file_path), img_dir=Path(cfg.igtd.img_dir)
            )
            train_idx, val_idx = trainer.k_fold_indices[0]
            dm.setup(train_idx=train_idx, val_idx=val_idx, test_idx=trainer.test_idx)
            y_test, y_pred = trainer.evaluate(dm.test_dataloader())

        case ModelType.CatBoost.value:
            trainer = EnsembleTreeTrainer(hyperparams=cfg.model.hyperparams)
            trainer.setup(data_cfg=cfg.data)
            trainer.cross_validate()

            logger.info("Evaluating the model on the test set ...")
            y_test, y_pred = trainer.evaluate()

            # Export predicted probabilities
            trainer.export_prob(output_path=Path(cfg.results.prob_path))

        case _:
            raise ValueError(f"Unsupported model type: {cfg.model.name}")

    # Compute metrics
    results = [compute_metrics(y_test, y_pred, avg) for avg in cfg.results.avg_options]

    # Export results
    results_df = pd.DataFrame(results)
    results_df.to_csv(cfg.results.file_path, index=False)
    logger.info(f"Predictions for the test set saved to {cfg.results.file_path}")


if __name__ == "__main__":
    main()
