import time
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchinfo import summary

from data.cdc import CDCDataModule
from data.transform import nctd_transform
from model.nctd import NCTDConvNet
from model.classifier import DiabetesRiskClassifier
from train_and_eval.evaluate import compute_metrics
from train_and_eval.metrics_logger import MetricsLogger


def main():
    torch.manual_seed(seed=42)

    # Init data module
    dm = CDCDataModule(
        data_file=Path("data/cdcNormalDiabetic.csv"),
        target_col="Label",
    )
    dm.setup(transform=nctd_transform, downsample=True)
    # dm.setup(transform=nctd_transform)

    # Init CNN model
    logger.info("Initializing NCTDConvNet model...")
    model = NCTDConvNet()
    summary(model, input_size=(1, 1, 42, 42))

    # dummy_input = torch.randn(8, 1, 42, 42)
    # output = model(dummy_input)

    # logger.debug(f"Output shape: {output.shape}")  # should be [8, 2]

    # dataloader = dm.train_dataloader()
    # batch = next(iter(dataloader))
    # inputs, labels = batch
    # logger.debug(f"Input shape: {inputs.shape}")
    # logger.debug(f"Label shape: {labels.shape}")

    logger.debug(f"MPS available: {torch.backends.mps.is_available()}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.debug(f"Current device: {device}")


    logger.info("Training the model...")
    metrics_logger = MetricsLogger()
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.001, patience=3, verbose=True, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # same metric as EarlyStopping
        mode="min",
        save_top_k=1,  # keep only the best model
        filename="best-checkpoint",
        verbose=False,
        save_weights_only=False,  # save full model (or True for just weights)
    )
    classifier = DiabetesRiskClassifier(model=model, batch_size=64)
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="mps",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=50,  # or higher
        enable_model_summary=False,  # optional
        # callbacks=[metrics_logger]
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

            logits = model(x)            # [B, 1]
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().squeeze()

            y_pred.extend(preds.cpu().numpy())
            y_test.extend(y.cpu().numpy())
    
    avg_options = ['micro', 'macro', 'weighted', 'binary']

    results = [compute_metrics(y_test, y_pred, avg) for avg in avg_options]
    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv", index=False)

# binary,0.8517226426994638,0.6291718170580964,0.12819544138017883,0.21299299089862955
# binary,0.8511313465783664,0.5647530040053405,0.21307140158670193,0.3094084300996617    
# binary,0.8138796909492274,0.4271156832298137,0.5542123158292407,0.482433543436558
# binary,0.7998265531378114,0.4038545012587898,0.5858204256390883,0.4781089414182939

if __name__ == "__main__":
    main()
