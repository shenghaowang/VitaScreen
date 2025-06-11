from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


class CDCDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        transform: Callable = None
    ):

        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        
        return x, y


class CDCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_file: Path,
        target_col: str,
        batch_size: int = 64
    ):
        super().__init__()
        self.data_file = data_file
        self.target_col = target_col
        self.batch_size = batch_size
    
    def setup(self, transform: Callable = None):
        # Load raw data
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file {self.data_file} does not exist.")
        
        df = pd.read_csv(self.data_file)
        feature_cols = [col for col in df.columns if col != self.target_col]
        
        # Split data for training, validation, and testing
        X, y = df[feature_cols].values, df[self.target_col].values
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42
        )

        # Scale the features
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Initialize datasets
        self.train_dataset = CDCDataset(X_train, y_train, transform)
        self.val_dataset = CDCDataset(X_val, y_val, transform)
        self.test_dataset = CDCDataset(X_test, y_test, transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)