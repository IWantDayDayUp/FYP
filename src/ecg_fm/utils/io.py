# src/ecg_fm/utils/io.py
from __future__ import annotations

import json
from pathlib import Path


def make_run_dir(out_root: Path, run_name: str) -> Path:
    run_dir = out_root / "runs" / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
