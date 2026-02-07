# src/ecg_fm/training/cls_baseline.py
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import classification_report, confusion_matrix, f1_score

from ecg_fm.utils.io import make_run_dir, save_json
from ecg_fm.utils.system import get_device_info, get_git_info

from ecg_fm.data.pairs_shard_dataset import PairsShardDataset
from ecg_fm.models.registry import build_model, list_models


# =====================================================================
# Argparser
# =====================================================================
def build_cls_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # dataset + splits
    p.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        help='List of task names, e.g. "mitdb_beat incartdb_beat qtdb_beat"',
    )
    p.add_argument("--pairs-train", type=str, required=True)
    p.add_argument("--pairs-val", type=str, required=True)
    p.add_argument("--pairs-test", type=str, required=True)

    p.add_argument(
        "--data-root",
        type=str,
        default=os.environ.get("FYP_DATA_DIR", ""),
        help="Root directory where shard npy files live",
    )
    p.add_argument("--out", type=str, default="./outputs", help="Output root directory")
    p.add_argument(
        "--run-name", type=str, default="", help="Run name (default timestamp)"
    )

    # ============ Aggregation level ============
    p.add_argument(
        "--agg-level",
        type=str,
        default="beat",
        choices=["beat", "window", "window-centered"],
        help="Aggregation level for predictions: "
        "beat (no agg), window (fixed window), window-centered (beat-centered window)",
    )
    p.add_argument(
        "--agg-window-size",
        type=int,
        default=30,
        help="Window size for window-level and window-centered aggregation",
    )

    # sampling strategy
    # Sampling strategy
    p.add_argument(
        "--sampling",
        type=str,
        default="none",
        choices=["none", "wrs", "oversample", "hybrid", "smote"],
        help="Train sampling strategy",
    )
    p.add_argument(
        "--sampling-replacement",
        action="store_true",
        help="Use replacement in WeightedRandomSampler",
    )
    p.add_argument(
        "--oversample-total-samples",
        type=int,
        default=None,
        help="Target total samples when oversampling (default: num_classes * max_count)",
    )
    p.add_argument(
        "--hybrid-ratio",
        type=float,
        default=0.5,
        help="Target ratio for hybrid sampling (0.0-1.0)",
    )

    # training
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=0)

    # model
    p.add_argument("--model", type=str, default="cnn_small")
    p.add_argument("--num-classes", type=int, default=5)
    p.add_argument("--input-len", type=int, default=300)

    # imbalance
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "wce", "focal"])
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--weights-alpha", type=float, default=0.5)
    p.add_argument("--label-smoothing", type=float, default=0.0)

    # FM
    p.add_argument("--fm-ckpt", type=str, default=None)
    p.add_argument("--fm-feature", type=str, default="mean", choices=["mean", "cls"])
    p.add_argument("--fm-unfreeze-last-n", type=int, default=0)

    return p


# =====================================================================
# Dataset helpers
# =====================================================================
def _load_xy_shards(data_root: Path, prefix: str):
    """Load shards for a single task."""
    shard_ids = sorted(
        {p.stem.split("_shard")[-1] for p in data_root.glob(f"{prefix}_X_shard*.npy")}
    )
    if len(shard_ids) == 0:
        raise FileNotFoundError(f"No shards found for {prefix} under {data_root}")

    Xs = [
        np.load(data_root / f"{prefix}_X_shard{sid}.npy", mmap_mode="r")
        for sid in shard_ids
    ]
    ys = [
        np.load(data_root / f"{prefix}_y_shard{sid}.npy", mmap_mode="r")
        for sid in shard_ids
    ]

    sizes = [len(a) for a in ys]
    offs = np.cumsum([0] + sizes)
    return Xs, ys, offs


# =====================================================================
# Training / Evaluation
# =====================================================================
@torch.no_grad()
def _eval_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str
):
    """Standard epoch evaluation (beat-level, no aggregation)."""
    model.eval()
    total_loss = 0.0
    n = 0
    ys_all, ps_all = [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += float(loss.item()) * len(yb)
        n += len(yb)

        pred = torch.argmax(logits, dim=1)
        ys_all.append(yb.cpu().numpy())
        ps_all.append(pred.cpu().numpy())

    ys_all = np.concatenate(ys_all)
    ps_all = np.concatenate(ps_all)
    avg_loss = total_loss / max(n, 1)
    acc = float((ys_all == ps_all).mean())
    return avg_loss, acc, ys_all, ps_all


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    opt,
    device: str,
    max_steps: int = 0,
):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    n = 0
    steps = 0

    for step, (xb, yb) in enumerate(loader, start=1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        opt.step()

        total_loss += float(loss.item()) * len(yb)
        n += len(yb)
        steps += 1

        if max_steps > 0 and step >= max_steps:
            break

    avg_loss = total_loss / max(n, 1)
    return avg_loss, steps


# =====================================================================
# Hybrid Sampling (Undersample + Oversample)
# =====================================================================
def apply_hybrid_sampling(
    train_labels: np.ndarray,
    train_counts: np.ndarray,
    args: argparse.Namespace,
    log,
) -> Tuple[WeightedRandomSampler, bool]:
    """
    Hybrid sampling: undersample majority + oversample minority to middle ground.

    Args:
        train_labels: [N] label array
        train_counts: [num_classes] class counts
        args: contains hybrid_ratio (0.0 to 1.0)
        log: logging function

    Returns:
        sampler, shuffle flag
    """
    log(f"\nSampling: Hybrid (undersample + oversample)")
    log(f"Class counts: {train_counts.tolist()}")
    log(f"Hybrid ratio: {args.hybrid_ratio}")

    min_count = float(train_counts.min())
    max_count = float(train_counts.max())

    # Target: min + ratio * (max - min)
    target_count = min_count + args.hybrid_ratio * (max_count - min_count)
    target_count = int(target_count)

    log(f"Min count: {min_count}, Max count: {max_count}")
    log(f"Target count per class: {target_count}")

    # Weights: min(1.0, target / actual_count)
    # - Majority classes: weight < 1.0 (undersample via probabilistic selection)
    # - Minority classes: weight = 1.0 (may oversample with replacement)
    w_samples = np.minimum(1.0, target_count / train_counts[train_labels]).astype(
        np.float64
    )
    w_samples = torch.from_numpy(w_samples)

    total_samples = int(args.num_classes * target_count)

    sampler = WeightedRandomSampler(
        weights=w_samples,
        num_samples=total_samples,
        replacement=True,
    )

    log(f"Total samples (after resampling): {total_samples}")
    return sampler, False


# =====================================================================
# SMOTE Sampling
# =====================================================================
class SMOTEDatasetWrapper(Dataset):
    """Wrapper for SMOTE-resampled data."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """
        Args:
            X: [N, input_len] feature array
            Y: [N] label array
        """
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return (x, y, task_id=0, local_idx=idx) for compatibility
        return self.X[idx], self.Y[idx], 0, idx


def _load_dataset_to_memory(
    dataset: "PairsShardDataset", args: argparse.Namespace, log
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load entire dataset into memory for SMOTE processing.

    Args:
        dataset: PairsShardDataset
        args: argparser namespace (contains input_len)
        log: logging function

    Returns:
        (X, Y) arrays
    """
    log(f"Loading {len(dataset)} samples into memory...")

    all_X = []
    all_Y = []

    for i in range(len(dataset)):
        try:
            x, y, _, _ = dataset[i]
            # x is torch tensor or numpy
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            all_X.append(x)
            all_Y.append(int(y))
        except Exception as e:
            log(f"  Warning: Skipped sample {i}: {e}")
            continue

    X = np.array(all_X, dtype=np.float32)  # [N, input_len]
    Y = np.array(all_Y, dtype=np.int64)  # [N]

    log(f"Loaded {len(X)} samples, shape: {X.shape}, Y shape: {Y.shape}")
    return X, Y


def apply_smote_sampling(
    dataset: "PairsShardDataset",
    train_counts: np.ndarray,
    args: argparse.Namespace,
    log,
) -> Tuple[SMOTEDatasetWrapper, bool]:
    """
    SMOTE: Synthetic Minority Over-sampling Technique.

    Generates synthetic samples for minority classes via k-NN interpolation.

    Args:
        dataset: Original PairsShardDataset
        train_counts: [num_classes] class counts
        args: argparser namespace
        log: logging function

    Returns:
        (new_dataset, shuffle_flag)
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        log("[ERROR] imbalanced-learn not installed!")
        log("  Install with: pip install imbalanced-learn")
        log("  Falling back to weighted random sampling")
        import sys

        return None, None  # Signal fallback

    log(f"\nSampling: SMOTE (Synthetic Minority Over-sampling)")
    log(f"Class counts: {train_counts.tolist()}")

    # Load all data into memory
    X, Y = _load_dataset_to_memory(dataset, args, log)

    log(f"Loaded {len(X)} samples for SMOTE processing")

    # Apply SMOTE
    try:
        # Resample all classes except majority to match majority
        smote = SMOTE(
            sampling_strategy="not majority",
            random_state=args.seed,
            k_neighbors=5,
        )

        log(f"Applying SMOTE...")
        X_resampled, Y_resampled = smote.fit_resample(X, Y)

        new_counts = np.bincount(Y_resampled, minlength=args.num_classes)
        log(f"After SMOTE: {len(X_resampled)} samples")
        log(f"New class distribution: {new_counts.tolist()}")

        # Create wrapped dataset
        new_dataset = SMOTEDatasetWrapper(X_resampled, Y_resampled)

        return new_dataset, True  # shuffle=True for new synthetic data

    except Exception as e:
        log(f"[ERROR] SMOTE failed: {e}")
        log("Falling back to weighted random sampling")
        return None, None  # Signal fallback


# =====================================================================
# Class Imbalance Handling
# =====================================================================
def compute_class_counts_from_pairs(
    ds: "PairsShardDataset", num_classes: int
) -> np.ndarray:
    """Fast class counting from pairs."""
    counts = np.zeros(num_classes, dtype=np.int64)

    for task_id, local_idx in ds.pairs:
        Xs, ys, offs = ds.task_xy[int(task_id)]

        if len(offs) == 2:
            shard_id = 0
            in_shard = int(local_idx)
        else:
            shard_id = int(np.searchsorted(offs, int(local_idx), side="right") - 1)
            in_shard = int(local_idx) - int(offs[shard_id])

        y = int(ys[shard_id][in_shard])
        if y < 0 or y >= num_classes:
            raise ValueError(f"Invalid label y={y}, expected [0, {num_classes-1}]")
        counts[y] += 1

    return counts


def compute_labels_from_pairs(ds: "PairsShardDataset", num_classes: int) -> np.ndarray:
    """Get label array aligned with ds.pairs."""
    labels = np.zeros(len(ds.pairs), dtype=np.int64)

    for i, (task_id, local_idx) in enumerate(ds.pairs):
        Xs, ys, offs = ds.task_xy[int(task_id)]

        if len(offs) == 2:
            shard_id = 0
            in_shard = int(local_idx)
        else:
            shard_id = int(np.searchsorted(offs, int(local_idx), side="right") - 1)
            in_shard = int(local_idx) - int(offs[shard_id])

        y = int(ys[shard_id][in_shard])
        if y < 0 or y >= num_classes:
            raise ValueError(f"Invalid label y={y}, expected [0, {num_classes-1}]")
        labels[i] = y

    return labels


def compute_tempered_class_weights(
    counts: np.ndarray,
    alpha: float = 0.5,
    eps: float = 1e-12,
    clip_min: float | None = 0.2,
    clip_max: float | None = 5.0,
) -> np.ndarray:
    """
    Compute class weights: w_c ∝ 1/(n_c^alpha).
    Normalize to mean=1, optional clip, then renormalize.
    """
    counts = counts.astype(np.float64)
    w = 1.0 / np.power(np.maximum(counts, 1.0), alpha)

    # 1) normalize first
    w = w / (w.mean() + eps)

    # 2) clip on normalized scale
    if clip_min is not None or clip_max is not None:
        lo = -np.inf if clip_min is None else clip_min
        hi = np.inf if clip_max is None else clip_max
        w = np.clip(w, lo, hi)

        # 3) renormalize
        w = w / (w.mean() + eps)

    return w


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            w = self.weight.gather(0, target)
            loss = loss * w
        return loss.mean()


def build_criterion(args, train_counts: np.ndarray, device: str):
    """Build loss function."""
    label_smoothing = float(getattr(args, "label_smoothing", 0.0))

    if args.loss == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing), None

    if args.loss == "wce":
        w_np = compute_tempered_class_weights(train_counts, alpha=args.weights_alpha)
        w = torch.tensor(w_np, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing), w_np

    if args.loss == "focal":
        w_np = compute_tempered_class_weights(train_counts, alpha=args.weights_alpha)
        w = torch.tensor(w_np, dtype=torch.float32, device=device)
        return FocalLoss(gamma=args.focal_gamma, weight=w), w_np

    raise ValueError(f"Unknown loss type: {args.loss}")


def _f1_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Dict[str, object]:
    """Compute macro/weighted F1 and per-class recall."""
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    accuracy = float((y_true == y_pred).mean())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    denom = cm.sum(axis=1)
    denom = np.clip(denom, 1, None)
    per_class_recall = (np.diag(cm) / denom).astype(np.float64)

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "per_class_recall": per_class_recall.tolist(),
    }


# =====================================================================
# Prediction with Aggregation
# =====================================================================
def _get_predictions(
    model: nn.Module,
    dataset: Dataset,
    device: str,
    agg_level: str = "beat",
    agg_window_size: int = 30,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions with optional aggregation.

    Args:
        model: trained model
        dataset: test/val dataset
        device: cuda or cpu
        agg_level: "beat" | "window" | "window-centered"
        agg_window_size: window size for aggregation
        batch_size: batch size for inference

    Returns:
        (y_pred, y_true): aggregated predictions and targets
    """
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)  # [B, num_classes]
            all_logits.append(logits.cpu())
            all_targets.append(yb)

    logits = torch.cat(all_logits, dim=0)  # [N, num_classes]
    targets = torch.cat(all_targets, dim=0)  # [N]
    preds = logits.argmax(dim=1).numpy()
    targets = targets.numpy()

    # Apply aggregation if needed
    if agg_level == "beat":
        # No aggregation
        return preds, targets

    elif agg_level == "window":
        # Fixed window aggregation
        return _aggregate_window_level(preds, targets, window_size=agg_window_size)

    elif agg_level == "window-centered":
        # Beat-centered window aggregation
        return _aggregate_window_centered(preds, targets, window_size=agg_window_size)

    else:
        raise ValueError(f"Unknown agg_level: {agg_level}")


def _aggregate_window_level(
    preds: np.ndarray, targets: np.ndarray, window_size: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fixed window aggregation: divide predictions into non-overlapping windows,
    apply majority voting within each window.
    """
    agg_preds = []
    agg_targets = []

    for i in range(0, len(preds), window_size):
        window_preds = preds[i : i + window_size]
        window_targets = targets[i : i + window_size]

        if len(window_preds) == 0:
            continue

        # Majority vote
        agg_pred = np.bincount(window_preds).argmax()
        agg_target = np.bincount(window_targets).argmax()

        agg_preds.append(agg_pred)
        agg_targets.append(agg_target)

    return np.array(agg_preds), np.array(agg_targets)


def _aggregate_window_centered(
    preds: np.ndarray, targets: np.ndarray, window_size: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Beat-centered window aggregation: for each beat, take a window of
    size (window_size//2) beats before and after, apply majority voting.
    """
    agg_preds = []
    agg_targets = []

    half_win = window_size // 2

    for i in range(len(preds)):
        # Center window around beat i
        start = max(0, i - half_win)
        end = min(len(preds), i + half_win + 1)

        window_preds = preds[start:end]
        window_targets = targets[start:end]

        # Majority vote
        agg_pred = np.bincount(window_preds).argmax()
        agg_target = np.bincount(window_targets).argmax()

        agg_preds.append(agg_pred)
        agg_targets.append(agg_target)

    return np.array(agg_preds), np.array(agg_targets)


# =====================================================================
# Main Training Function
# =====================================================================
def train_classifier(args: argparse.Namespace) -> None:
    """Main training and evaluation pipeline."""

    # ========== Setup ==========
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or time.strftime("%Y-%m-%d_%H%M_cls")
    run_dir = make_run_dir(out_root, run_name)

    log_path = run_dir / "train.log"
    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.json"
    meta_path = run_dir / "meta.json"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        """Log to both console and file."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    # SLURM info
    job_id = os.environ.get("SLURM_JOB_ID", "N/A")
    slurm_info = {
        "slurm_job_id": job_id,
        "slurm_submit_dir": os.environ.get("SLURM_SUBMIT_DIR", ""),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "slurm_nnodes": os.environ.get("SLURM_JOB_NUM_NODES", ""),
        "slurm_ntasks": os.environ.get("SLURM_NTASKS", ""),
        "slurm_cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE", ""),
        "slurm_gres": os.environ.get("SLURM_JOB_GRES", ""),
    }

    device_info = get_device_info(device)
    git_info = get_git_info()

    data_root = Path(args.data_root) if args.data_root else Path(".")
    tasks: List[str] = args.tasks

    log(f"{'='*70}")
    log(f"Training Classification Model")
    log(f"{'='*70}")
    log(f"Run directory: {run_dir}")
    log(f"Aggregation level: {args.agg_level} (window_size={args.agg_window_size})")
    log(f"Sampling strategy: {args.sampling}")
    log(f"Loss function: {args.loss}")
    log(f"Model: {args.model}")
    log(f"Device: {device}")

    # ========== Load Data ==========
    pairs_train = np.load(args.pairs_train)
    pairs_val = np.load(args.pairs_val)
    pairs_test = np.load(args.pairs_test)

    ds_train = PairsShardDataset(pairs_train, tasks, data_root, args.input_len)
    ds_val = PairsShardDataset(pairs_val, tasks, data_root, args.input_len)
    ds_test = PairsShardDataset(pairs_test, tasks, data_root, args.input_len)

    log(f"\nDataset sizes: train={len(ds_train)} val={len(ds_val)} test={len(ds_test)}")

    # ========== Class Counts & Loss ==========
    train_counts = compute_class_counts_from_pairs(
        ds_train, num_classes=args.num_classes
    )
    log(f"Train class counts: {train_counts.tolist()}")

    criterion, w_np = build_criterion(args, train_counts, device)
    log(f"Loss: {args.loss} (alpha={args.weights_alpha}, gamma={args.focal_gamma})")
    if w_np is not None:
        log(f"Class weights: {np.round(w_np, 6).tolist()}")

    # ========== Build Sampler ==========
    sampler = None
    shuffle = True

    log(f"\n{'='*70}")
    log(f"Building Sampler (strategy: {args.sampling})")
    log(f"{'='*70}")

    if args.sampling == "none":
        log(f"Sampling: None (standard shuffling)")
        shuffle = True

    elif args.sampling == "wrs":
        # Weighted Random Sampler
        train_labels = compute_labels_from_pairs(ds_train, num_classes=args.num_classes)
        w_cls = compute_tempered_class_weights(train_counts, alpha=args.weights_alpha)
        log(f"\nSampling: WeightedRandomSampler (alpha={args.weights_alpha})")
        log(f"Class weights: {np.round(w_cls, 6).tolist()}")

        w_samples = w_cls[train_labels].astype(np.float64)
        w_samples = torch.from_numpy(w_samples)

        sampler = WeightedRandomSampler(
            weights=w_samples,
            num_samples=len(ds_train),
            replacement=bool(args.sampling_replacement),
        )
        shuffle = False

    elif args.sampling == "oversample":
        # Random Oversampling
        train_labels = compute_labels_from_pairs(ds_train, args.num_classes)
        max_count = train_counts.max()
        log(f"\nSampling: Random Oversampling")
        log(f"Max class count: {max_count}")
        log(f"Class counts: {train_counts.tolist()}")

        w_samples = (max_count / train_counts[train_labels]).astype(np.float64)
        w_samples = torch.from_numpy(w_samples)

        total_samples = int(
            getattr(args, "oversample_total_samples", None)
            or (int(max_count) * int(args.num_classes))
        )

        sampler = WeightedRandomSampler(
            weights=w_samples,
            num_samples=total_samples,
            replacement=True,
        )
        shuffle = False
        log(f"Total samples (with replacement): {total_samples}")

    elif args.sampling == "hybrid":
        # Hybrid: undersample + oversample
        train_labels = compute_labels_from_pairs(ds_train, args.num_classes)
        sampler, shuffle = apply_hybrid_sampling(train_labels, train_counts, args, log)

    elif args.sampling == "smote":
        # SMOTE: generate synthetic minority samples
        result = apply_smote_sampling(ds_train, train_counts, args, log)

        if result[0] is None:
            # Fallback to weighted random sampling
            log(f"\n[FALLBACK] Using WeightedRandomSampler instead")
            train_labels = compute_labels_from_pairs(ds_train, args.num_classes)
            w_cls = compute_tempered_class_weights(
                train_counts, alpha=args.weights_alpha
            )
            w_samples = w_cls[train_labels].astype(np.float64)
            w_samples = torch.from_numpy(w_samples)

            sampler = WeightedRandomSampler(
                weights=w_samples,
                num_samples=len(ds_train),
                replacement=True,
            )
            shuffle = False
        else:
            # SMOTE succeeded: use new dataset and shuffle
            ds_train, shuffle = result
            sampler = None
            log(f"✅ SMOTE applied, using new dataset with {len(ds_train)} samples")

    else:
        raise ValueError(f"Unknown sampling strategy: {args.sampling}")

    # ========== DataLoaders ==========
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else 2,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    log(f"Tasks: {tasks}")
    log(f"Data root: {data_root}")

    # ========== Model ==========
    model = build_model(
        args.model,
        **vars(args),
    ).to(device)

    log(f"\nAvailable models: {list_models()}")
    log(f"Model: {args.model}")
    log(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # ========== Optimizer ==========
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    log(f"Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})")

    # ========== Save Meta ==========
    save_json(
        meta_path,
        {
            "args": vars(args),
            "slurm": slurm_info,
            "device": device_info,
            "git": git_info,
            "data": {
                "tasks": tasks,
                "data_root": str(data_root),
                "pairs_train": str(args.pairs_train),
                "pairs_val": str(args.pairs_val),
                "pairs_test": str(args.pairs_test),
                "n_train": len(ds_train),
                "n_val": len(ds_val),
                "n_test": len(ds_test),
                "train_class_counts": train_counts.tolist(),
            },
        },
    )

    # ========== Metrics CSV ==========
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_acc",
                "val_macro_f1",
                "val_weighted_f1",
                "val_recall_c0",
                "val_recall_c1",
                "val_recall_c2",
                "val_recall_c3",
                "val_recall_c4",
                "epoch_sec",
                "steps",
                "gpu_mem_alloc_mb",
            ]
        )

    # ========== Training Loop ==========
    # use macro F1 on validation to select best model (more relevant for imbalance)
    best_val_metric = -1.0
    best_epoch = -1
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    t0 = time.time()

    log(f"\n{'='*70}")
    log(f"Starting training for {args.epochs} epochs")
    log(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Train
        train_loss, steps = _train_epoch(
            model, dl_train, criterion, opt, device, max_steps=args.max_steps
        )

        # Validation (beat-level, no aggregation)
        val_loss, val_acc, yv, pv = _eval_epoch(model, dl_val, criterion, device)
        val_m = _f1_metrics(yv, pv, num_classes=args.num_classes)
        val_macro_f1 = val_m["macro_f1"]
        val_weighted_f1 = val_m["weighted_f1"]
        val_recalls = val_m["per_class_recall"]

        epoch_sec = time.time() - epoch_start
        gpu_alloc = (
            int(torch.cuda.max_memory_allocated() / 1024 / 1024)
            if device == "cuda"
            else 0
        )

        log(
            f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"val_macro_f1={val_macro_f1:.4f} | val_weighted_f1={val_weighted_f1:.4f}"
        )

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_acc:.6f}",
                    f"{val_macro_f1:.6f}",
                    f"{val_weighted_f1:.6f}",
                    f"{val_recalls[0]:.6f}",
                    f"{val_recalls[1]:.6f}",
                    f"{val_recalls[2]:.6f}",
                    f"{val_recalls[3]:.6f}",
                    f"{val_recalls[4]:.6f}",
                    f"{epoch_sec:.3f}",
                    steps,
                    gpu_alloc,
                ]
            )

        # Save checkpoints
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "args": vars(args),
            },
            last_path,
        )

        # select best by validation macro F1
        if val_macro_f1 > best_val_metric:
            best_val_metric = val_macro_f1
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_val_metric": best_val_metric,
                    "args": vars(args),
                },
                best_path,
            )
            log(
                f"  → Best model saved by val_macro_f1={best_val_metric:.4f} (epoch {epoch})"
            )

    elapsed_sec = time.time() - t0
    log(f"\n{'='*70}")
    log(f"Training completed in {elapsed_sec:.1f}s")
    log(f"Best epoch: {best_epoch}")
    log(f"{'='*70}\n")

    # ========== Load Best Model & Test ==========
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    log(f"Loaded best model from epoch {best_epoch}")

    log(f"\n{'='*70}")
    log(f"Test Evaluation with agg_level={args.agg_level}")
    log(f"{'='*70}\n")

    # 【关键】获取测试集预测（应用指定的聚合级别）
    y_pred_test, y_true_test = _get_predictions(
        model=model,
        dataset=ds_test,
        device=device,
        agg_level=args.agg_level,
        agg_window_size=args.agg_window_size,
        batch_size=args.batch_size,
    )

    # 计算测试指标
    test_m = _f1_metrics(y_true_test, y_pred_test, num_classes=args.num_classes)
    log(f"[TEST] macro_f1={test_m['macro_f1']:.4f}")
    log(f"[TEST] weighted_f1={test_m['weighted_f1']:.4f}")
    log(f"[TEST] accuracy={test_m['accuracy']:.4f}")
    log(
        f"[TEST] per_class_recall={np.array(test_m['per_class_recall']).round(4).tolist()}"
    )

    # Classification report
    report = classification_report(
        y_true_test, y_pred_test, output_dict=True, zero_division=0
    )
    report_path = run_dir / "test_classification_report.json"
    save_json(report_path, report)

    # Confusion matrix
    cm = confusion_matrix(y_true_test, y_pred_test)
    cm_path = run_dir / "confusion_matrix.csv"
    np.savetxt(cm_path, cm, fmt="%d", delimiter=",")

    log(f"\nConfusion matrix shape: {cm.shape}")
    log(
        f"\nClassification Report:\n{classification_report(y_true_test, y_pred_test, zero_division=0)}"
    )

    # ========== 【关键】Save Summary with Aggregation Info ==========
    # actual evaluated test sample count may differ after aggregation
    eval_n = int(len(y_true_test))

    summary = {
        "job_id": job_id,
        "device": device,
        "best_epoch": best_epoch,
        "elapsed_sec": float(elapsed_sec),
        "run_dir": str(run_dir),
        # 【关键】Aggregation info
        "agg_level": args.agg_level,
        "agg_window_size": args.agg_window_size,
        "test_size": len(ds_test),
        "test_evaluated_samples": eval_n,
        # Sampling & Loss
        "sampling": args.sampling,
        "loss": args.loss,
        "weights_alpha": args.weights_alpha,
        "focal_gamma": args.focal_gamma,
        # Test metrics
        "test_macro_f1": float(test_m["macro_f1"]),
        "test_weighted_f1": float(test_m["weighted_f1"]),
        "test_accuracy": float(test_m["accuracy"]),
        "test_per_class_recall": test_m["per_class_recall"],
        # File paths
        "paths": {
            "meta": str(meta_path),
            "metrics": str(metrics_path),
            "confusion_matrix": str(cm_path),
            "log": str(log_path),
            "ckpt_best": str(best_path),
            "ckpt_last": str(last_path),
            "report": str(report_path),
        },
    }
    save_json(summary_path, summary)

    log(f"\n✅ [DONE] Results saved to {run_dir}")
    log(f"   Summary: {summary_path}")
    log(f"   Log: {log_path}")
    log(f"   Metrics: {metrics_path}")
