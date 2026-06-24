"""
src/dataset.py
PyTorch Dataset and DataLoader construction for apple classification.

Expected directory layout (ImageFolder-compatible):
    data/raw/
        ├── class_a/
        │     ├── img001.jpg
        │     └── ...
        └── class_b/
              └── ...

Run as a script to generate train/val/test split CSVs:
    python src/dataset.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from utils import get_logger, load_config, ensure_dirs

logger = get_logger(__name__)


# ── Transforms ────────────────────────────────────────────────────────────


def build_transforms(cfg: DictConfig, split: str = "train") -> A.Compose:
    """Return an albumentations pipeline for *split* ∈ {train, val, test}."""
    aug = cfg.augmentation
    size = cfg.data.image_size
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if split == "train":
        transforms = [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5 if aug.train.horizontal_flip else 0.0),
            A.VerticalFlip(p=0.5 if aug.train.vertical_flip else 0.0),
            A.Rotate(limit=aug.train.random_rotation, p=0.5),
        ]
        if aug.train.color_jitter:
            transforms.append(
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)
            )
        if aug.train.random_erasing:
            transforms.append(A.CoarseDropout(p=0.3))
        transforms += [A.Normalize(mean=mean, std=std), ToTensorV2()]
    else:
        transforms = [A.Resize(size, size), A.Normalize(mean=mean, std=std), ToTensorV2()]

    return A.Compose(transforms)


# ── Dataset ───────────────────────────────────────────────────────────────


class AppleDataset(Dataset):
    """
    Loads images listed in a CSV manifest.

    CSV format (no header required, just these two columns):
        filepath,label
        data/raw/class_a/img001.jpg,0
        ...
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        transform: A.Compose | None = None,
        root: str | Path = ".",
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform
        self.root = Path(root)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple:
        row = self.manifest.iloc[idx]
        img_path = self.root / row["filepath"]
        label = int(row["label"])

        image = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label


# ── DataLoader factory ────────────────────────────────────────────────────


def build_loaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Build train / val / test DataLoaders from split CSVs.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    splits_dir = Path(cfg.data.splits)
    class_names = _load_class_names(splits_dir)

    def _loader(split: str, shuffle: bool) -> DataLoader:
        df = pd.read_csv(splits_dir / f"{split}.csv")
        transform = build_transforms(cfg, split)
        ds = AppleDataset(df, transform=transform, root=".")
        return DataLoader(
            ds,
            batch_size=cfg.training.batch_size,
            shuffle=shuffle,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
        )

    return (
        _loader("train", shuffle=True),
        _loader("val", shuffle=False),
        _loader("test", shuffle=False),
        class_names,
    )


def _load_class_names(splits_dir: Path) -> list[str]:
    names_file = splits_dir / "classes.txt"
    if names_file.exists():
        return names_file.read_text().strip().splitlines()
    return []


# ── Split builder (run as script) ─────────────────────────────────────────


def build_splits(cfg: DictConfig) -> None:
    """
    Scan cfg.data.root for an ImageFolder layout and produce
    train/val/test CSVs + classes.txt in cfg.data.splits.
    """
    root = Path(cfg.data.root)
    splits_dir = Path(cfg.data.splits)
    ensure_dirs(splits_dir)

    records: list[dict] = []
    class_names: list[str] = sorted(p.name for p in root.iterdir() if p.is_dir())

    for label, cls in enumerate(class_names):
        for img_path in (root / cls).rglob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
                records.append({"filepath": str(img_path), "label": label})

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Total images found: {len(df)}")
    logger.info(f"Classes: {class_names}")

    # 70 / 15 / 15 split
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)
    (splits_dir / "classes.txt").write_text("\n".join(class_names))

    logger.info(f"Splits saved to {splits_dir}  "
                f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})")


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dataset splits")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    build_splits(cfg)
