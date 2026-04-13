"""
Time-series classification dataset loader.

Supports loading from UCR/UEA archive format (tsv/arff) or pre-processed
numpy arrays. Patches the time series for LLM input.
"""

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class TSClassificationDataset(Dataset):
    """Time-series classification dataset.

    Each sample: (patches, label).
    The full time series is split into patches for the LLM backbone.

    Args:
        data: Numpy array (N, T, D) — N samples, T timesteps, D features.
        labels: Numpy array (N,) — integer class labels.
        patch_size: Patch size for tokenisation.
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray, patch_size: int = 16):
        self.data = data
        self.labels = labels
        self.patch_size = patch_size
        self.n_features = data.shape[2] if data.ndim == 3 else 1
        self.seq_len = data.shape[1]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]  # (T, D) or (T,)
        if sample.ndim == 1:
            sample = sample[:, None]

        # Truncate to multiple of patch_size
        T = (self.seq_len // self.patch_size) * self.patch_size
        sample = sample[:T]
        n_patches = T // self.patch_size

        patches = sample.reshape(n_patches, self.patch_size, self.n_features)
        patches = patches.reshape(n_patches, self.patch_size * self.n_features)

        return {
            "patches": torch.tensor(patches, dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "target": torch.tensor(sample, dtype=torch.float32),  # dummy for compatibility
            "mask": torch.ones(T, self.n_features, dtype=torch.float32),  # dummy
        }


def load_ucr_dataset(
    name: str = "ECG200",
    data_dir: str = "data/classification",
) -> dict:
    """Load a UCR dataset from tsv format.

    Expects: {data_dir}/{name}/{name}_TRAIN.tsv and {name}_TEST.tsv
    UCR format: first column is label, remaining columns are time series values.
    """
    train_path = os.path.join(data_dir, name, f"{name}_TRAIN.tsv")
    test_path = os.path.join(data_dir, name, f"{name}_TEST.tsv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"UCR dataset not found at {train_path}. "
            f"Download from https://www.timeseriesclassification.com/ "
            f"and extract to {data_dir}/{name}/"
        )

    train_raw = np.loadtxt(train_path, delimiter="\t")
    test_raw = np.loadtxt(test_path, delimiter="\t")

    train_labels = train_raw[:, 0].astype(int)
    train_data = train_raw[:, 1:]
    test_labels = test_raw[:, 0].astype(int)
    test_data = test_raw[:, 1:]

    # Remap labels to 0-indexed
    unique_labels = np.unique(np.concatenate([train_labels, test_labels]))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    train_labels = np.array([label_map[l] for l in train_labels])
    test_labels = np.array([label_map[l] for l in test_labels])

    # Normalise with train stats
    mean = train_data.mean()
    std = train_data.std() + 1e-8
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    # Handle NaN
    train_data = np.nan_to_num(train_data, 0.0)
    test_data = np.nan_to_num(test_data, 0.0)

    # Reshape to (N, T, 1) for univariate
    train_data = train_data[:, :, None].astype(np.float32)
    test_data = test_data[:, :, None].astype(np.float32)

    # Split train into train/val (80/20)
    n = len(train_data)
    val_start = int(n * 0.8)

    return {
        "train_data": train_data[:val_start],
        "train_labels": train_labels[:val_start],
        "val_data": train_data[val_start:],
        "val_labels": train_labels[val_start:],
        "test_data": test_data,
        "test_labels": test_labels,
        "n_features": 1,
        "n_classes": len(unique_labels),
        "seq_len": train_data.shape[1],
        "name": name,
    }


def create_classification_datasets(
    data_dict: dict,
    patch_size: int = 16,
) -> Tuple[TSClassificationDataset, TSClassificationDataset, TSClassificationDataset]:
    """Create train/val/test classification datasets."""
    train_ds = TSClassificationDataset(data_dict["train_data"], data_dict["train_labels"], patch_size)
    val_ds = TSClassificationDataset(data_dict["val_data"], data_dict["val_labels"], patch_size)
    test_ds = TSClassificationDataset(data_dict["test_data"], data_dict["test_labels"], patch_size)
    return train_ds, val_ds, test_ds
