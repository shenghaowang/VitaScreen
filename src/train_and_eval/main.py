from pathlib import Path

import hydra
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data.cdc import IgtdDataModule, NctdDataModule
from data.transform import nctd_transform
from model.model_type import ModelType
from train_and_eval.convnet_trainer import ConvNetTrainer
from train_and_eval.ensemble_tree_trainer import EnsembleTreeTrainer
from train_and_eval.evaluate import compute_metrics


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    torch.manual_seed(seed=42)

    # Init data module
    match cfg.model.name:
        case ModelType.NCTD.value:
            dm = NctdDataModule(
                data_file=Path(cfg.data.file_path),
                target_col=cfg.data.target_col,
            )
            dm.setup(transform=nctd_transform, downsample=cfg.data.downsample)

        case ModelType.IGTD.value:
            dm = IgtdDataModule(
                data_file=Path(cfg.data.file_path),
                img_dir=Path(cfg.igtd.img_dir),
                target_col=cfg.data.target_col,
            )
            dm.setup()

        case ModelType.CatBoost.value:
            # No datamodule needed for CatBoost
            dm = None

        case _:
            raise ValueError(f"Unsupported model type: {cfg.model.name}")

    if cfg.model.name in (ModelType.NCTD.value, ModelType.IGTD.value):
        trainer = ConvNetTrainer(
            data_module=dm,
            model_cfg=cfg.model,
            train_cfg=cfg.train,
        )
        trainer.init_trainer()
        trainer.train()
        trainer.test()

    else:
        trainer = EnsembleTreeTrainer(hyperparams=cfg.model.hyperparams)
        trainer.setup(data_cfg=cfg.data)
        trainer.train()

    # Evaluate the model
    logger.info("Evaluating the model...")
    y_test, y_pred = trainer.evaluate()
    results = [compute_metrics(y_test, y_pred, avg) for avg in cfg.results.avg_options]

    # Export results
    results_df = pd.DataFrame(results)
    results_df.to_csv(cfg.results.file_path, index=False)


if __name__ == "__main__":
    main()
