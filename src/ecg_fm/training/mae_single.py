# src/ecg_fm/training/mae_single.py
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ecg_fm.data.datasets import SingleNpyECGDataset
from ecg_fm.models.mini_mae import ECGMAE_1D, mae_loss_masked
from ecg_fm.utils.io import make_run_dir, save_json
from ecg_fm.utils.system import get_device_info, get_git_info, bytes_to_mb


def build_mae_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npy",
        type=str,
        required=True,
        help="Path to .npy (or relative to --data-root)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.environ.get("FYP_DATA_DIR", ""),
        help="Optional data root",
    )
    parser.add_argument(
        "--out", type=str, default="./outputs", help="Output root directory"
    )
    parser.add_argument(
        "--run-name", type=str, default="", help="Run name (default auto timestamp)"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of samples (0 = no limit)"
    )
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--patch-size", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dim-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="If >0, run at most this many steps per epoch",
    )
    return parser


def train_mae_single(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or time.strftime("%Y-%m-%d_%H%M_mae")
    run_dir = make_run_dir(out_root, run_name)

    log_path = run_dir / "train.log"
    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.json"
    meta_path = run_dir / "meta.json"

    # resolve npy path
    npy_path = Path(args.npy)
    if args.data_root:
        npy_path = Path(args.data_root) / npy_path

    # logging helper
    def log(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

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

    # dataset
    limit = args.limit if args.limit > 0 else None
    ds = SingleNpyECGDataset(npy_path, limit=limit)
    log(f"Dataset length: {len(ds)}")

    # infer L from one sample
    x0 = ds[0]  # [1, L]
    L = int(x0.shape[-1])

    # data meta
    npy_size = npy_path.stat().st_size if npy_path.exists() else 0
    data_info = {
        "npy_path": str(npy_path),
        "npy_size_mb": round(bytes_to_mb(npy_size), 2),
        "num_samples": int(len(ds)),
        "signal_length_L": L,
        "batch_size": args.batch_size,
        "steps_per_epoch": int(len(ds) // args.batch_size),
    }

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    # model
    model = ECGMAE_1D(
        n_leads=1,
        d_model=args.d_model,
        patch_size=args.patch_size,
        enc_depth=4,
        dec_depth=2,
        n_heads=args.n_heads,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio,
        pos_max_len=4096,
    ).to(device)
    log(f"Model: {model}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # save immutable meta
    save_json(
        meta_path,
        {
            "args": vars(args),
            "slurm": slurm_info,
            "device": device_info,
            "git": git_info,
            "data": data_info,
        },
    )

    log(
        f"device={device} job_id={job_id} "
        f"partition={slurm_info['slurm_partition']} "
        f"gres={slurm_info['slurm_gres']}"
    )
    log(
        f"npy={npy_path} (size={data_info['npy_size_mb']} MB), "
        f"N={data_info['num_samples']}, L={L}"
    )
    log(f"patch_size={args.patch_size} (L % patch_size = {L % args.patch_size})")
    if L % args.patch_size != 0:
        log(
            "WARNING: L is not divisible by patch_size. "
            "Training will fail unless you changed model/padding."
        )

    # init metrics csv
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "avg_loss",
                "epoch_sec",
                "steps",
                "samples_per_sec",
                "gpu_mem_alloc_mb",
                "gpu_mem_reserved_mb",
            ]
        )

    best_loss = float("inf")
    global_step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        epoch_start = time.time()

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for step, x in enumerate(dl, start=1):
            x = x.to(device, non_blocking=True)  # [B, 1, L]

            pred, tgt, mask = model(x)
            loss = mae_loss_masked(pred, tgt, mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item()
            count += 1
            global_step += 1

            if args.max_steps > 0 and step >= args.max_steps:
                break

        epoch_sec = time.time() - epoch_start
        avg_loss = total / max(count, 1)

        samples = count * args.batch_size
        samples_per_sec = samples / max(epoch_sec, 1e-9)

        if device == "cuda":
            gpu_alloc = int(torch.cuda.memory_allocated() / 1024 / 1024)
            gpu_reserved = int(torch.cuda.memory_reserved() / 1024 / 1024)
            gpu_peak = int(torch.cuda.max_memory_allocated() / 1024 / 1024)
            log(
                f"Epoch {epoch}/{args.epochs} loss={avg_loss:.6f} "
                f"sec={epoch_sec:.2f} samples/s={samples_per_sec:.1f} "
                f"gpu_alloc={gpu_alloc}MB peak={gpu_peak}MB"
            )
        else:
            gpu_alloc = 0
            gpu_reserved = 0
            log(
                f"Epoch {epoch}/{args.epochs} loss={avg_loss:.6f} "
                f"sec={epoch_sec:.2f} samples/s={samples_per_sec:.1f}"
            )

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{avg_loss:.6f}",
                    f"{epoch_sec:.3f}",
                    count,
                    f"{samples_per_sec:.3f}",
                    gpu_alloc,
                    gpu_reserved,
                ]
            )

        # save last checkpoint
        ckpt_last = run_dir / "checkpoints" / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "best_loss": best_loss,
                "args": vars(args),
                "L": L,
            },
            ckpt_last,
        )

        # save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_best = run_dir / "checkpoints" / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "best_loss": best_loss,
                    "args": vars(args),
                    "L": L,
                },
                ckpt_best,
            )

    summary = {
        "job_id": job_id,
        "device": device,
        "best_loss": best_loss,
        "epochs_ran": args.epochs,
        "global_step": global_step,
        "elapsed_sec": time.time() - t0,
        "run_dir": str(run_dir),
        "paths": {
            "meta": str(meta_path),
            "metrics": str(metrics_path),
            "log": str(log_path),
            "ckpt_last": str(run_dir / "checkpoints" / "last.pt"),
            "ckpt_best": str(run_dir / "checkpoints" / "best.pt"),
        },
    }
    save_json(summary_path, summary)
    log("[Done] Finished training and saved checkpoints/metrics/meta/summary.")
