"""
Forecasting dataset: lookback window → predict future horizon.

Standard ETT forecasting protocol:
  - Short-term:  lookback=96, horizon=96
  - Medium-term: lookback=96, horizon=192
  - Long-term:   lookback=96, horizon=336

The model sees the lookback window as input patches and predicts the
horizon as output patches. No masking (all input values are observed).
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ForecastingDataset(Dataset):
    """Sliding-window forecasting dataset.

    Each sample: (lookback_patches, horizon_target).
    The model receives lookback_patches and predicts horizon_target.

    Args:
        data: Numpy array (T, D) — normalised time series.
        lookback: Length of the input window.
        horizon: Length of the prediction horizon.
        patch_size: Patch size for tokenisation.
        stride: Sliding window stride.
    """

    def __init__(
        self,
        data: np.ndarray,
        lookback: int = 96,
        horizon: int = 96,
        patch_size: int = 16,
        stride: int = 1,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.patch_size = patch_size
        self.n_features = data.shape[1]
        self.total_len = lookback + horizon

        self.samples = []
        for start in range(0, len(data) - self.total_len + 1, stride):
            self.samples.append(data[start : start + self.total_len])
        self.samples = np.stack(self.samples)  # (N, lookback+horizon, D)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        lookback_data = sample[:self.lookback]      # (lookback, D)
        horizon_data = sample[self.lookback:]        # (horizon, D)

        # Create patches from lookback
        n_patches = self.lookback // self.patch_size
        patches = lookback_data[:n_patches * self.patch_size].reshape(
            n_patches, self.patch_size, self.n_features
        ).reshape(n_patches, self.patch_size * self.n_features)

        return {
            "patches": torch.tensor(patches, dtype=torch.float32),
            "target": torch.tensor(horizon_data, dtype=torch.float32),
            "mask": torch.ones_like(torch.tensor(lookback_data, dtype=torch.float32)),  # dummy for compatibility
        }


def load_ett_forecasting(
    variant: str = "h1",
    horizon: int = 96,
    lookback: int = 96,
    data_dir: str = "data/time_series/ETDataset/ETT-small",
) -> dict:
    """Load ETT for forecasting with standard splits."""
    filename = f"ETT{variant}.csv"
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)
    data = df.iloc[:, 1:].values.astype(np.float32)

    # Standard ETT splits: 60/20/20
    n = len(data)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    # Normalise with train stats
    train_data = data[:train_end]
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0) + 1e-8

    return {
        "train": (data[:train_end] - mean) / std,
        "val": (data[train_end:val_end] - mean) / std,
        "test": (data[val_end:] - mean) / std,
        "n_features": data.shape[1],
        "name": f"ETT-{variant}",
        "horizon": horizon,
        "lookback": lookback,
        "mean": mean,
        "std": std,
    }


def create_forecasting_datasets(
    data_dict: dict,
    patch_size: int = 16,
    stride: int = 1,
) -> Tuple[ForecastingDataset, ForecastingDataset, ForecastingDataset]:
    """Create train/val/test forecasting datasets."""
    lookback = data_dict["lookback"]
    horizon = data_dict["horizon"]

    datasets = {}
    for split in ["train", "val", "test"]:
        datasets[split] = ForecastingDataset(
            data=data_dict[split],
            lookback=lookback,
            horizon=horizon,
            patch_size=patch_size,
            stride=stride if split == "train" else lookback,
        )

    return datasets["train"], datasets["val"], datasets["test"]
