# src/ecg_fm/utils/system.py
from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from typing import Dict, Any

import torch


def get_git_info() -> dict:
    """Best-effort git metadata. Empty strings if not a git repo."""

    def run(cmd: list[str]) -> str:
        try:
            out = (
                subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            )
            return out
        except Exception:
            return ""

    return {
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(run(["git", "status", "--porcelain"])),
    }


def get_device_info(device: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }

    if device == "cuda" and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "cuda_device_index": idx,
                "cuda_device_name": torch.cuda.get_device_name(idx),
                "cuda_runtime_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_mem_total_mb": int(props.total_memory / 1024 / 1024),
            }
        )
    return info


def bytes_to_mb(x: int) -> float:
    return x / 1024.0 / 1024.0
