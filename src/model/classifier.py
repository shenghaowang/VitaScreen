import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class DiabetesRiskClassifier(pl.LightningModule):
    def __init__(self, model):
        super(DiabetesRiskClassifier, self).__init__()
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.f1 = torchmetrics.classification.BinaryF1Score()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x).squeeze()  # Remove extra dimensions
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_f1', self.f1(y_hat, y.float()))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x).squeeze() 
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_f1', self.f1(y_hat, y.float()))
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x).squeeze()
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_f1', self.f1(y_hat, y.float()))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=8e-4)
