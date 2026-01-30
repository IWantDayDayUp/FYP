# src/ecg_fm/models/registry.py
from __future__ import annotations

from typing import Callable, Dict
import torch.nn as nn


_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    """
    Decorator to register a model builder function.

    Example:
        @register_model("cnn_small")
        def build_cnn_small(num_classes: int, **kwargs) -> nn.Module:
            return CNN1D_Small(num_classes=num_classes)
    """
    name = name.strip()

    def _wrap(fn: Callable[..., nn.Module]):
        if name in _MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' is already registered.")
        _MODEL_REGISTRY[name] = fn
        return fn

    return _wrap


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY.keys())


def build_model(name: str, **kwargs) -> nn.Module:
    """
    Build a model by name using kwargs.

    Raises:
        KeyError if name is not registered.
    """
    name = name.strip()
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list_models()}")
    return _MODEL_REGISTRY[name](**kwargs)
