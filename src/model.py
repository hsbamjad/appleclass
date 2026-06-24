"""
src/model.py
Model factory that wraps any timm backbone for apple classification.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig

from utils import get_logger

logger = get_logger(__name__)


class AppleClassifier(nn.Module):
    """
    Thin wrapper around a timm backbone with a custom classification head.

    Args:
        cfg: Full project config. Reads from cfg.model.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            cfg.model.name,
            pretrained=cfg.model.pretrained,
            num_classes=0,
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=cfg.model.dropout),
            nn.Linear(in_features, cfg.model.num_classes),
        )
        logger.info(
            "Model: %s | backbone features: %d | output classes: %d",
            cfg.model.name,
            in_features,
            cfg.model.num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_model(cfg: DictConfig) -> AppleClassifier:
    """Instantiate and return the model described in cfg."""
    return AppleClassifier(cfg)
