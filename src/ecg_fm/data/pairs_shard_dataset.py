# src/ecg_fm/data/pairs_shard_dataset.py

from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_xy_shards(data_root: Path, prefix: str):
    shard_ids = sorted(
        {p.stem.split("_shard")[-1] for p in data_root.glob(f"{prefix}_X_shard*.npy")}
    )
    if len(shard_ids) == 0:
        raise FileNotFoundError(f"No shards found for {prefix}")

    Xs = [
        np.load(data_root / f"{prefix}_X_shard{sid}.npy", mmap_mode="r")
        for sid in shard_ids
    ]
    ys = [
        np.load(data_root / f"{prefix}_y_shard{sid}.npy", mmap_mode="r")
        for sid in shard_ids
    ]
    sizes = [len(y) for y in ys]
    offs = np.cumsum([0] + sizes)
    return Xs, ys, offs


class PairsShardDataset(Dataset):
    """
    Dataset backed by multiple (X_shard, y_shard) npy files
    accessed via (task_id, local_index) pairs.
    """

    def __init__(
        self,
        pairs: np.ndarray,
        tasks: List[str],
        data_root: Path,
        input_len: int,
    ):
        self.pairs = pairs.astype(np.int64)
        self.tasks = tasks
        self.data_root = data_root
        self.input_len = input_len

        self.task_xy = []
        for t in tasks:
            self.task_xy.append(_load_xy_shards(data_root, t))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        task_id, local_idx = self.pairs[i]
        Xs, ys, offs = self.task_xy[int(task_id)]

        if len(offs) == 2:
            shard_id = 0
            in_shard = int(local_idx)
        else:
            shard_id = int(np.searchsorted(offs, int(local_idx), side="right") - 1)
            in_shard = int(local_idx) - int(offs[shard_id])

        x = np.array(Xs[shard_id][in_shard], dtype=np.float32)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        y = int(ys[shard_id][in_shard])

        return (
            torch.from_numpy(x).unsqueeze(0),  # [1, L]
            torch.tensor(y).long(),
        )
