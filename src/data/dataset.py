"""
Unified time-series dataset with MCAR masking and patch tokenization.

Loads CSV time-series data, applies missing-completely-at-random (MCAR)
masking, splits into fixed-length windows, and creates patches suitable
for frozen-LLM input.
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesImputationDataset(Dataset):
    """Windowed time-series dataset with MCAR masking for imputation.

    Each sample is a window of length `window_size` with `n_features` channels.
    Missing values are introduced via MCAR at the specified `missing_rate`.
    The model receives the masked input; the target is the original values
    at masked positions.

    Args:
        data: Numpy array of shape (T, D) — full time series.
        window_size: Length of each sliding window.
        patch_size: Size of each patch for LLM tokenization.
        missing_rate: Fraction of values to mask (MCAR).
        stride: Sliding window stride (default = window_size, no overlap).
        seed: Random seed for reproducible masking.
    """

    def __init__(
        self,
        data: np.ndarray,
        window_size: int = 96,
        patch_size: int = 16,
        missing_rate: float = 0.3,
        stride: Optional[int] = None,
        seed: int = 42,
    ):
        self.window_size = window_size
        self.patch_size = patch_size
        self.missing_rate = missing_rate
        self.stride = stride or window_size
        self.n_features = data.shape[1]
        self.seed = seed

        # Normalize per-feature (train stats should be computed externally for splits)
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0) + 1e-8
        self.data = (data - self.mean) / self.std

        # Create windows
        self.windows = []
        for start in range(0, len(self.data) - window_size + 1, self.stride):
            self.windows.append(self.data[start : start + window_size])
        self.windows = np.stack(self.windows)  # (N, W, D)

        # Pre-generate masks (deterministic per seed)
        rng = np.random.RandomState(seed)
        self.masks = rng.random(self.windows.shape) > missing_rate  # True = observed
        self.masks = self.masks.astype(np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        window = self.windows[idx]  # (W, D)
        mask = self.masks[idx]      # (W, D) — 1=observed, 0=missing
        masked_input = window * mask  # zero out missing values

        # Create patches: (W, D) -> (n_patches, patch_size * D)
        n_patches = self.window_size // self.patch_size
        patches = masked_input.reshape(n_patches, self.patch_size, self.n_features)
        patches = patches.reshape(n_patches, self.patch_size * self.n_features)

        return {
            "patches": torch.tensor(patches, dtype=torch.float32),
            "target": torch.tensor(window, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "masked_input": torch.tensor(masked_input, dtype=torch.float32),
        }


def load_ett(variant: str = "h1", data_dir: str = "data/time_series/ETDataset/ETT-small") -> dict:
    """Load ETT dataset and return train/val/test splits.

    Standard split: 60% train, 20% val, 20% test (following ETT convention).
    """
    filename = f"ETT{variant}.csv"
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)

    # Drop date column, keep numerical features
    data = df.iloc[:, 1:].values.astype(np.float32)  # (T, 7)

    # Standard ETT splits
    n = len(data)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    return {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test": data[val_end:],
        "n_features": data.shape[1],
        "name": f"ETT-{variant}",
    }


def load_csv_dataset(
    name: str,
    path: str,
    has_date_col: bool = True,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    max_features: Optional[int] = None,
) -> dict:
    """Generic loader for CSV time-series datasets.

    Args:
        name: Dataset display name.
        path: Path to CSV file.
        has_date_col: If True, skip the first column (date/timestamp).
        train_ratio: Fraction for training split.
        val_ratio: Fraction for validation split.
        max_features: If set, take only the first N features (for high-dim datasets).
    """
    df = pd.read_csv(path)
    start_col = 1 if has_date_col else 0
    data = df.iloc[:, start_col:].values.astype(np.float32)

    if max_features is not None and data.shape[1] > max_features:
        data = data[:, :max_features]

    # Handle NaN/inf in raw data
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test": data[val_end:],
        "n_features": data.shape[1],
        "name": name,
    }


# ── Convenience loaders per dataset ──

def load_weather(data_dir: str = "data/time_series") -> dict:
    return load_csv_dataset("Weather", os.path.join(data_dir, "weather.csv"),
                            has_date_col=True)

def load_electricity(data_dir: str = "data/time_series", max_features: int = 20) -> dict:
    """Electricity: 321 features, subsample to max_features for tractability."""
    return load_csv_dataset("Electricity", os.path.join(data_dir, "electricity.csv"),
                            has_date_col=False, max_features=max_features)

def load_exchange(data_dir: str = "data/time_series") -> dict:
    return load_csv_dataset("Exchange", os.path.join(data_dir, "exchange_rate.csv"),
                            has_date_col=True)

def load_traffic(data_dir: str = "data/time_series", max_features: int = 20) -> dict:
    """Traffic: 862 features, subsample to max_features for tractability."""
    return load_csv_dataset("Traffic", os.path.join(data_dir, "traffic.csv"),
                            has_date_col=True, max_features=max_features)

def load_ili(data_dir: str = "data/time_series") -> dict:
    return load_csv_dataset("ILI", os.path.join(data_dir, "national_illness.csv"),
                            has_date_col=True)


def create_datasets(
    data_dict: dict,
    window_size: int = 96,
    patch_size: int = 16,
    missing_rate: float = 0.3,
    seed: int = 42,
) -> Tuple[TimeSeriesImputationDataset, TimeSeriesImputationDataset, TimeSeriesImputationDataset]:
    """Create train/val/test datasets from a data dict (from load_ett etc.)."""
    # Compute normalization stats from training data only
    train_mean = data_dict["train"].mean(axis=0)
    train_std = data_dict["train"].std(axis=0) + 1e-8

    datasets = {}
    for split in ["train", "val", "test"]:
        # Normalize using TRAIN stats
        normalized = (data_dict[split] - train_mean) / train_std
        ds = TimeSeriesImputationDataset(
            data=data_dict[split],  # raw data, will be re-normalized inside
            window_size=window_size,
            patch_size=patch_size,
            missing_rate=missing_rate,
            seed=seed + (0 if split == "train" else 1 if split == "val" else 2),
        )
        # Override with train-normalized data
        ds.mean = train_mean
        ds.std = train_std
        ds.data = normalized
        # Rebuild windows
        ds.windows = []
        for start in range(0, len(normalized) - window_size + 1, ds.stride):
            ds.windows.append(normalized[start : start + window_size])
        ds.windows = np.stack(ds.windows)
        # Rebuild masks
        rng = np.random.RandomState(seed + (0 if split == "train" else 1 if split == "val" else 2))
        ds.masks = (rng.random(ds.windows.shape) > missing_rate).astype(np.float32)
        datasets[split] = ds

    return datasets["train"], datasets["val"], datasets["test"]
