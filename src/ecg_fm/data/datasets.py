# data/dataset.py
from __future__ import annotations
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset
from typing import Optional, List


class SingleNpyECGDataset(Dataset):
    """
    Simple dataset: one .npy file -> one Dataset.

    Expected .npy shape: [N, L]
    Output: x of shape [1, L] with dtype float32.
    """

    def __init__(self, npy_path: str | Path, limit: int | None = None):
        super().__init__()
        self.npy_path = Path(npy_path)
        if not self.npy_path.exists():
            raise FileNotFoundError(f"npy not found: {self.npy_path}")

        # Use memory mapping to avoid loading the entire array into RAM
        self.data = np.load(self.npy_path, mmap_mode="r")
        if self.data.ndim != 2:
            raise ValueError(f"Expect 2D array [N, L], got {self.data.shape}")

        # Treat 0 / None as "no limit"
        if limit is None or limit == 0:
            self.N = int(self.data.shape[0])
        else:
            self.N = min(int(limit), int(self.data.shape[0]))

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.data[idx]  # [L]
        x = torch.from_numpy(x.copy()).float()  # [L] float32
        return x.unsqueeze(0)  # [1, L]


def build_multidb_dataset(
    npy_paths: List[str | Path],
    data_root: str | Path | None = None,
    limit_per_db: Optional[int] = None,
) -> ConcatDataset:
    """
    Build a concatenated dataset from multiple .npy files or directories.

    Each element in `npy_paths` can be:
      - a single .npy file
      - a directory containing multiple .npy shards
    """
    root = Path(data_root) if data_root else None
    datasets: list[Dataset] = []

    all_files: list[Path] = []

    for p in npy_paths:
        p = Path(p)
        # prepend data_root if given and path is not absolute
        if root is not None and not p.is_absolute():
            p = root / p

        if p.is_dir():
            # use all .npy files in this directory
            files = sorted(p.glob("*.npy"))
            all_files.extend(files)
        else:
            all_files.append(p)

    if not all_files:
        raise ValueError("No .npy files found for MultiDB dataset")

    for f in all_files:
        ds = SingleNpyECGDataset(f, limit=limit_per_db)
        datasets.append(ds)

    return ConcatDataset(datasets)
