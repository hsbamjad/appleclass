"""
src/evaluate.py
Evaluate a trained checkpoint on the test split and save a confusion matrix.

Usage:
    python src/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

from dataset import build_loaders
from model import build_model
from utils import ensure_dirs, get_device, get_logger, load_checkpoint, load_config

logger = get_logger(__name__)


def evaluate(cfg, checkpoint_path: str) -> dict:
    device = get_device()
    ensure_dirs(cfg.paths.figure_dir)

    _, _, test_loader, class_names = build_loaders(cfg)

    model = build_model(cfg).to(device)
    state = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(state["model"])
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    report = classification_report(
        all_labels, all_preds,
        target_names=class_names if class_names else None,
        digits=4,
    )
    logger.info(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names or range(cm.shape[1]),
        yticklabels=class_names or range(cm.shape[0]),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    fig_path = Path(cfg.paths.figure_dir) / "confusion_matrix.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    logger.info(f"Confusion matrix saved to {fig_path}")

    accuracy = (all_preds == all_labels).mean()
    return {"accuracy": float(accuracy), "report": report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate apple classifier on test set")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = evaluate(cfg, args.checkpoint)
    logger.info(f"Test accuracy: {results['accuracy']:.4f}")
