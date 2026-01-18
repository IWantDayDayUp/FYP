# src/data/downstream_dataset.py
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


class NpyECGDataset(Dataset):
    """
    Load ECG windows (or beats) from .npy files.

    X:
      - shape [N, L]  (single-lead) OR
      - shape [N, C, L] (multi-lead)

    y:
      - shape [N]
    """

    def __init__(self, x_path: str, y_path: str, dtype=np.float32):
        self.X = np.load(x_path).astype(dtype)
        self.y = np.load(y_path).astype(np.int64)

        assert len(self.X) == len(self.y), "X and y must have same length"

        # Normalize shape to [N, C, L]
        if self.X.ndim == 2:
            self.X = self.X[:, None, :]  # [N, 1, L]
        elif self.X.ndim == 3:
            pass
        else:
            raise ValueError(f"Unsupported X shape: {self.X.shape}")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])  # [C, L]
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
