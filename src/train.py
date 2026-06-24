"""
src/train.py
Main training loop for apple classification.

Usage:
    python src/train.py --config configs/default.yaml [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset import build_loaders
from model import build_model
from utils import (
    ensure_dirs,
    get_device,
    get_logger,
    load_config,
    save_checkpoint,
    seed_everything,
)

logger = get_logger(__name__)


# Optimizer and scheduler factories


def build_optimizer(cfg, params):
    """Return the configured optimizer."""
    name = cfg.training.optimizer.lower()
    lr = cfg.training.learning_rate
    wd = cfg.training.weight_decay
    if name == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=wd)
    return AdamW(params, lr=lr, weight_decay=wd)


def build_scheduler(cfg, optimizer):
    """Return the configured learning-rate scheduler."""
    name = cfg.training.scheduler.lower()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    return StepLR(optimizer, step_size=10, gamma=0.5)


# Single-epoch pass


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler,
    mixed_precision: bool,
    split: str = "train",
) -> tuple[float, float]:
    """Run one epoch of training or evaluation. Returns (mean_loss, accuracy)."""
    training = split == "train"
    model.train(training)

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc=split, leave=False):
            images, labels = images.to(device), labels.to(device)
            with torch.autocast("cuda", enabled=mixed_precision and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                if mixed_precision and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


# Main training entry point


def train(cfg) -> None:
    """Run the full training loop."""
    seed_everything(42)
    device = get_device()
    logger.info("Device: %s", device)

    ensure_dirs(cfg.paths.log_dir, cfg.paths.figure_dir, cfg.paths.checkpoint_dir)

    train_loader, val_loader, _, class_names = build_loaders(cfg)
    logger.info("Classes: %s", class_names)

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg, model.parameters())
    scheduler = build_scheduler(cfg, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.mixed_precision)

    best_val_acc = 0.0
    patience_counter = 0
    ckpt_dir = Path(cfg.paths.checkpoint_dir)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": cfg.model.name,
                "pretrained": cfg.model.pretrained,
                "epochs": cfg.training.epochs,
                "batch_size": cfg.training.batch_size,
                "lr": cfg.training.learning_rate,
                "optimizer": cfg.training.optimizer,
                "scheduler": cfg.training.scheduler,
            }
        )

        for epoch in range(1, cfg.training.epochs + 1):
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer,
                device, scaler, cfg.training.mixed_precision, "train",
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, None,
                device, scaler, False, "val",
            )

            if cfg.training.scheduler == "plateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()

            logger.info(
                "Epoch %03d/%d  train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f",
                epoch, cfg.training.epochs, train_loss, train_acc, val_loss, val_acc,
            )
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                path=ckpt_dir / f"epoch_{epoch:03d}.pt",
                is_best=is_best,
                best_path=ckpt_dir / "best.pt",
            )

            if patience_counter >= cfg.training.early_stopping_patience:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

        mlflow.log_metric("best_val_acc", best_val_acc)

    logger.info("Training complete. Best val_acc = %.4f", best_val_acc)


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train apple classifier")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(load_config(args.config))
