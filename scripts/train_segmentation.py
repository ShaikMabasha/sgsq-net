"""
Train the U-Net segmentation model (Stage 2) on the synthetic or real
segmentation dataset.

Usage:
    python scripts/train_segmentation.py --config configs/config.yaml
    python scripts/train_segmentation.py --config configs/config.yaml --epochs 2   # smoke test
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.dataset import SegmentationDataset
from models.unet import ResNetUNet
from utils.early_stopping import EarlyStopping


def compute_positive_sample_weights(seg_root: str, filenames: list) -> list:
    """
    Returns per-sample weights for WeightedRandomSampler such that, in
    expectation, positive (has-target-pixels) and negative (empty) images
    are drawn EQUALLY OFTEN per epoch, regardless of the underlying class
    imbalance in the raw file counts.
    """
    masks_dir = Path(seg_root) / "masks"
    is_positive = []
    for fname in filenames:
        stem = Path(fname).stem
        mask_path = masks_dir / f"{stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        is_positive.append(bool(mask is not None and (mask > 0).any()))

    n_pos = sum(is_positive)
    n_neg = len(is_positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        print("[warn] train split has no examples of one class -- skipping weighted sampling")
        return None

    weights = [(1.0 / n_pos) if pos else (1.0 / n_neg) for pos in is_positive]
    print(f"Weighted sampling enabled: {n_pos} positive / {n_neg} negative train images "
          f"-> resampled to ~50/50 per epoch in expectation.")
    return weights


def per_sample_dice(logits, targets, threshold: float = 0.5, smooth: float = 1.0):
    """Returns per-sample Dice scores (not batch-averaged), plus a bool tensor
    of which samples have any positive ground-truth pixel.

    Splitting by this flag is critical on imbalanced data: with Dice
    smoothing, correctly predicting an all-background mask on an
    all-background target scores ~1.0 for FREE, with zero real segmentation
    skill. Reporting Dice split by has_positive exposes this immediately.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    dice = (2 * intersection + smooth) / (preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + smooth)
    has_positive = targets_flat.sum(dim=1) > 0
    return dice, has_positive


def dice_score(logits, targets, threshold: float = 0.5, smooth: float = 1.0):
    """Batch-averaged Dice (legacy interface, kept for other scripts that import this)."""
    dice, _ = per_sample_dice(logits, targets, threshold, smooth)
    return dice.mean().item()


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        return bce_loss + (1 - dice.mean())


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    all_dice_sum, all_dice_n = 0.0, 0
    pos_dice_sum, pos_dice_n = 0.0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, masks, _ in tqdm(loader, desc="train" if train else "val", leave=False):
            images, masks = images.to(device), masks.to(device)
            if train:
                optimizer.zero_grad()

            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            loss = criterion(logits, masks)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            n += images.size(0)

            per_sample, has_positive = per_sample_dice(logits, masks)
            all_dice_sum += per_sample.sum().item()
            all_dice_n += per_sample.size(0)
            if has_positive.any():
                pos_dice_sum += per_sample[has_positive].sum().item()
                pos_dice_n += has_positive.sum().item()

    avg_loss = total_loss / n
    dice_all = all_dice_sum / all_dice_n
    dice_positive_only = (pos_dice_sum / pos_dice_n) if pos_dice_n > 0 else None
    positive_fraction = pos_dice_n / all_dice_n if all_dice_n > 0 else 0.0
    return avg_loss, dice_all, dice_positive_only, positive_fraction


def main(config_path: str, epochs_override: int = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    seg_cfg = cfg["segmentation"]
    epochs = epochs_override or seg_cfg["epochs"]

    seg_root = cfg["paths"]["segmentation_dataset"]
    all_images_dir = Path(seg_root) / "images"
    train_files = sorted([p.name for p in all_images_dir.glob("*.jpg") if p.name.startswith("train_")])
    val_files = sorted([p.name for p in all_images_dir.glob("*.jpg") if p.name.startswith("val_")])

    if len(train_files) == 0:
        raise FileNotFoundError(
            f"No images found under {all_images_dir} — run the appropriate converter/generator first."
        )

    train_ds = SegmentationDataset(seg_root, filenames=train_files, img_size=seg_cfg["img_size"], train=True)
    val_ds = SegmentationDataset(seg_root, filenames=val_files, img_size=seg_cfg["img_size"], train=False)

    sample_weights = compute_positive_sample_weights(seg_root, train_files)
    if sample_weights is not None:
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_files), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=seg_cfg["batch_size"], sampler=sampler, num_workers=2)
    else:
        train_loader = DataLoader(train_ds, batch_size=seg_cfg["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=seg_cfg["batch_size"], shuffle=False, num_workers=2)

    model = ResNetUNet(num_classes=seg_cfg["num_classes"]).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=seg_cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_dice = 0.0
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Select the best checkpoint by POSITIVE-ONLY Dice, not overall Dice.
    early_stopper = EarlyStopping(patience=seg_cfg.get("early_stopping_patience", 15), mode="max")

    for epoch in range(epochs):
        train_loss, train_dice, train_dice_pos, train_pos_frac = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_dice, val_dice_pos, val_pos_frac = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        val_pos_str = f"{val_dice_pos:.4f}" if val_dice_pos is not None else "N/A"
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} | "
            f"val_dice_POSITIVE_ONLY={val_pos_str} (frac_positive_val={val_pos_frac:.2f})"
        )

        selection_metric = val_dice_pos if val_dice_pos is not None else val_dice
        is_best = early_stopper.step(selection_metric)
        if is_best:
            best_dice = selection_metric
            torch.save(model.state_dict(), ckpt_dir / "segmentation_best.pt")
            print(f"  -> saved new best checkpoint (val_dice_positive_only={selection_metric:.4f})")

        if early_stopper.should_stop:
            print(f"  -> early stopping triggered (no improvement for {early_stopper.patience} epochs)")
            break

    print(f"\nBest validation Dice (positive-only): {best_dice:.4f} (stopped after {epoch+1}/{epochs} epochs)")
    return best_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.epochs)
