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
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix, f1_score

from ecg_fm.utils.io import make_run_dir, save_json
from ecg_fm.utils.system import get_device_info, get_git_info

from ecg_fm.data.pairs_shard_dataset import PairsShardDataset
from ecg_fm.models.registry import build_model, list_models


# -----------------------------
# Argparser
# -----------------------------
def build_cls_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # dataset + splits
    p.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        help='List of task names, e.g. "mitdb_beat incartdb_beat qtdb_beat sddb_beat"',
    )
    p.add_argument(
        "--pairs-train",
        type=str,
        required=True,
        help="Path to pairs_train.npy (each row: [task_id, local_index])",
    )
    p.add_argument("--pairs-val", type=str, required=True)
    p.add_argument("--pairs-test", type=str, required=True)

    p.add_argument(
        "--data-root",
        type=str,
        default=os.environ.get("FYP_DATA_DIR", ""),
        help="Root directory where shard npy files live (can use $FYP_DATA_DIR).",
    )
    p.add_argument("--out", type=str, default="./outputs", help="Output root directory")
    p.add_argument(
        "--run-name", type=str, default="", help="Run name (default timestamp)"
    )

    # training
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-steps", type=int, default=0, help="If >0, limit steps per epoch"
    )

    # model
    p.add_argument(
        "--model",
        type=str,
        default="cnn_small",
        help="Model name (see ecg_fm.models.list_models())",
    )

    p.add_argument("--num-classes", type=int, default=5)
    p.add_argument("--input-len", type=int, default=300)

    # imbalance (baseline default: none)
    p.add_argument(
        "--use-class-weights", action="store_true", help="Enable weighted CE"
    )
    p.add_argument(
        "--weights-alpha",
        type=float,
        default=0.5,
        help="Tempered weights exponent (0.5 is mild)",
    )
    p.add_argument("--label-smoothing", type=float, default=0.0)

    # FM
    p.add_argument("--fm-ckpt", type=str, default=None, help="Path to FM checkpoint")
    p.add_argument(
        "--fm-feature",
        type=str,
        default="mean",
        choices=["mean", "cls"],
        help="How to pool FM features",
    )
    p.add_argument(
        "--fm-unfreeze-last-n",
        type=int,
        default=0,
        help="For finetune: unfreeze last N blocks",
    )

    return p


# -----------------------------
# Dataset (manifest pairs -> memmap shards)
# -----------------------------
def _load_xy_shards(data_root: Path, prefix: str):
    # Find shard ids by scanning X files
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
    offs = np.cumsum([0] + sizes)  # prefix sums, len = n_shards+1
    return Xs, ys, offs


# def build_model(name: str, num_classes: int) -> nn.Module:
#     if name == "cnn_small":
#         return CNN1D_Small(num_classes)
#     raise ValueError(f"Unknown model: {name}")


# -----------------------------
# Training / eval
# -----------------------------
@torch.no_grad()
def _eval_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str
):
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


def _compute_tempered_weights(counts: np.ndarray, alpha: float) -> torch.Tensor:
    # weight_c = (N / count_c)^alpha, then normalize to mean=1
    counts = counts.astype(np.float64)
    counts[counts == 0] = 1.0
    w = (counts.sum() / counts) ** alpha
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def _f1_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Dict[str, object]:
    """
    Return macro/weighted F1 + per-class recall.
    """
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # per-class recall = TP / (TP + FN)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    denom = cm.sum(axis=1)  # true counts per class
    denom = np.clip(denom, 1, None)
    per_class_recall = (np.diag(cm) / denom).astype(np.float64)

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_recall": per_class_recall.tolist(),
    }


def train_classifier(args: argparse.Namespace) -> None:
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
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    # SLURM info (optional)
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

    # Load pairs
    pairs_train = np.load(args.pairs_train)
    pairs_val = np.load(args.pairs_val)
    pairs_test = np.load(args.pairs_test)

    # Dataset / Loader
    ds_train = PairsShardDataset(pairs_train, tasks, data_root, args.input_len)
    ds_val = PairsShardDataset(pairs_val, tasks, data_root, args.input_len)
    ds_test = PairsShardDataset(pairs_test, tasks, data_root, args.input_len)

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
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

    log(f"Train/Val/Test lens: {len(ds_train)} / {len(ds_val)} / {len(ds_test)}")
    log(f"Tasks: {tasks}")
    log(f"Data root: {data_root}")

    # Model
    model = build_model(
        args.model,
        # num_classes=args.num_classes,
        **vars(args),
    ).to(device)

    log(f"Available models: {list_models()}")
    log(f"Using model: {args.model}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Compute class weights from TRAIN (fast: just count y by iterating once)
    class_weights = None
    if args.use_class_weights:
        counts = np.zeros(args.num_classes, dtype=np.int64)
        for _, yb in dl_train:
            y_np = yb.numpy()
            counts += np.bincount(y_np, minlength=args.num_classes)
        w = _compute_tempered_weights(counts, alpha=args.weights_alpha)
        class_weights = w.to(device)
        log(f"Train class counts: {counts.tolist()}")
        log(
            f"Class weights (alpha={args.weights_alpha}): {w.cpu().numpy().round(4).tolist()}"
        )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(args.label_smoothing),
    )

    # Save meta (immutable)
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
            },
        },
    )

    # Metrics CSV
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

    best_val = float("inf")
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        train_loss, steps = _train_epoch(
            model, dl_train, criterion, opt, device, max_steps=args.max_steps
        )
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
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"val_macro_f1={val_macro_f1:.4f} | val_weighted_f1={val_weighted_f1:.4f} | "
            f"val_recall={np.array(val_recalls).round(4).tolist()}"
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

        # save last
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "args": vars(args),
            },
            last_path,
        )

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_val": best_val,
                    "args": vars(args),
                },
                best_path,
            )

    # Final test using best ckpt
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_loss, test_acc, yt, pt = _eval_epoch(model, dl_test, criterion, device)
    log(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")

    test_m = _f1_metrics(yt, pt, num_classes=args.num_classes)
    log(
        f"[TEST] macro_f1={test_m['macro_f1']:.4f} weighted_f1={test_m['weighted_f1']:.4f} "
        f"per_class_recall={np.array(test_m['per_class_recall']).round(4).tolist()}"
    )

    # Save reports
    report = classification_report(yt, pt, output_dict=True, zero_division=0)
    report_path = run_dir / "test_classification_report.json"
    save_json(report_path, report)

    cm = confusion_matrix(yt, pt)
    np.savetxt(run_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    summary = {
        "job_id": job_id,
        "device": device,
        "best_val_loss": best_val,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "elapsed_sec": time.time() - t0,
        "run_dir": str(run_dir),
        "paths": {
            "meta": str(meta_path),
            "metrics": str(metrics_path),
            "confusion_matrix": str(run_dir / "confusion_matrix.csv"),
            "log": str(log_path),
            "ckpt_best": str(best_path),
            "ckpt_last": str(last_path),
            "report": str(report_path),
        },
        "test_macro_f1": float(test_m["macro_f1"]),
        "test_weighted_f1": float(test_m["weighted_f1"]),
        "test_per_class_recall": test_m["per_class_recall"],
    }
    save_json(summary_path, summary)
    log(
        "[Done] Finished classification training and saved metrics/checkpoints/reports."
    )
