from pathlib import Path

import hydra
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from model.model_type import ModelType
from train_and_eval.ensemble_tree_trainer import EnsembleTreeTrainer
from train_and_eval.evaluate import compute_metrics
from train_and_eval.neuralnet_trainer import NeuralNetTrainer


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    torch.manual_seed(seed=42)

    # Data splitting is now handled within each trainer's setup() method

    # Initialize trainer based on model type - all use cross-validation
    match cfg.model.name:
        case ModelType.MLP.value | ModelType.NCTD.value | ModelType.IGTD.value:
            trainer = NeuralNetTrainer(
                model_cfg=cfg.model,
                train_cfg=cfg.train,
            )
            trainer.setup(data_cfg=cfg.data)
            trainer.cross_validate(data_file=Path(cfg.data.file_path))

        case ModelType.CatBoost.value:
            trainer = EnsembleTreeTrainer(hyperparams=cfg.model.hyperparams)
            trainer.setup(data_cfg=cfg.data)
            trainer.cross_validate()

        case _:
            raise ValueError(f"Unsupported model type: {cfg.model.name}")

    # Evaluate the model
    logger.info("Evaluating the model...")
    y_test, y_pred = trainer.evaluate()
    results = [compute_metrics(y_test, y_pred, avg) for avg in cfg.results.avg_options]

    # Export results
    results_df = pd.DataFrame(results)
    results_df.to_csv(cfg.results.file_path, index=False)
    logger.info(f"Predictions for the test set saved to {cfg.results.file_path}")


if __name__ == "__main__":
    main()
