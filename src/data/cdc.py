from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from imblearn.under_sampling import EditedNearestNeighbours
from loguru import logger
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


def prepare_data(
    data_file: Path, target_col: str, downsample: bool = False
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Load and prepare data for training."""

    # Load raw data
    df = pd.read_csv(data_file)
    feature_cols = [col for col in df.columns if col != target_col]

    # Split data for training, validation, and testing
    X, y = df[feature_cols].values, df[target_col].values
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if downsample:
        # Resampling using Edited Nearest Neighbours
        enn = EditedNearestNeighbours()
        X_train_val, y_train_val = enn.fit_resample(X_train_val, y_train_val)
        logger.info(
            f"Resampled training data shape: {X_train_val.shape}, {y_train_val.shape}"
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class NctdCDCDataset(Dataset):
    def __init__(
        self, data: np.ndarray, labels: np.ndarray, transform: Callable = None
    ):

        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        n_features = len(self.data[idx])

        if self.transform:
            x = self.transform(self.data[idx], n_features)

        # logger.debug(f"Transformed data shape: {x.shape}")

        # Add channel dimension for CNNs
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        y = torch.tensor(self.labels[idx], dtype=torch.float32)

        return x, y


class IgtdCDCDataset(Dataset):
    def __init__(self, img_dir: Path, labels: pd.Series):
        self.img_dir = img_dir
        self.img_files = sorted(img_dir.glob("*_image.png"))

        # Extract indices from filenames like _123_image.png
        self.indices = [int(f.name.split("_")[1]) for f in self.img_files]

        logger.debug(f"Found {len(self.img_files)} images in {self.img_dir}")
        logger.debug(f"Number of labels: {len(labels)}")
        assert (
            len(labels) >= max(self.indices) + 1
        ), "Label series does not cover all image indices"

        self.labels = labels
        self.transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        index = self.indices[idx]
        x = self.transform(Image.open(self.img_files[idx]))
        y = torch.tensor(self.labels.iloc[index], dtype=torch.float32)
        return x, y


class CDCDataModule(pl.LightningDataModule, ABC):
    def __init__(
        self,
        data_file: Path,
        target_col: str,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_file = data_file
        self.target_col = target_col
        self.batch_size = batch_size
        self.num_workers = num_workers

    @abstractmethod
    def setup(self, stage: str = None, **kwargs):
        raise NotImplementedError(
            "The setup method must be implemented in the subclass."
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class NctdDataModule(CDCDataModule):
    def __init__(
        self,
        data_file: Path,
        target_col: str,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__(
            data_file=data_file,
            target_col=target_col,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def setup(self, transform: Callable = None, downsample: bool = False):
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file {self.data_file} does not exist.")

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(
            data_file=self.data_file,
            target_col=self.target_col,
            downsample=downsample,
        )

        # Scale the features
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Initialize datasets
        self.train_dataset = NctdCDCDataset(X_train, y_train, transform)
        self.val_dataset = NctdCDCDataset(X_val, y_val, transform)
        self.test_dataset = NctdCDCDataset(X_test, y_test, transform)


class IgtdDataModule(CDCDataModule):
    def __init__(
        self,
        data_file: Path,
        img_dir: Path,
        target_col: str = "Label",
        batch_size=64,
        num_workers=4,
    ):
        super().__init__(
            data_file=data_file,
            target_col=target_col,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.img_dir = img_dir

    def setup(self):
        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"IGTD data directory {self.img_dir} does not exist."
            )

        df = pd.read_csv(self.data_file)
        train_val_idx, test_idx = train_test_split(
            df.index, test_size=0.1, random_state=42
        )
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.2, random_state=42
        )

        full_dataset = IgtdCDCDataset(img_dir=self.img_dir, labels=df[self.target_col])
        self.train_dataset = Subset(full_dataset, train_idx)
        self.val_dataset = Subset(full_dataset, val_idx)
        self.test_dataset = Subset(full_dataset, test_idx)
