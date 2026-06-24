"""
src/utils.py
Common utility helpers shared across the pipeline.
"""
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf


# ── Logging ───────────────────────────────────────────────────────────────


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Config ────────────────────────────────────────────────────────────────


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML config file and return an OmegaConf DictConfig."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return OmegaConf.create(raw)


# ── Reproducibility ───────────────────────────────────────────────────────


def seed_everything(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── Device ────────────────────────────────────────────────────────────────


def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Directory helpers ─────────────────────────────────────────────────────


def ensure_dirs(*paths: str | Path) -> None:
    """Create directories (including parents) if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# ── Checkpoint helpers ────────────────────────────────────────────────────


def save_checkpoint(
    state: dict[str, Any],
    path: str | Path,
    is_best: bool = False,
    best_path: str | Path | None = None,
) -> None:
    """Save a training checkpoint; optionally also save as *best*."""
    torch.save(state, path)
    if is_best and best_path is not None:
        import shutil
        shutil.copyfile(path, best_path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint from *path* onto *device*."""
    return torch.load(path, map_location=device)
