"""
Train baseline end-to-end severity classifiers directly on raw images
(no segmentation stage). These serve as:

  - BENCHMARK comparisons against literature architectures
  - The ABLATION baseline: "does segmentation-guided severity actually help?"

BUG FIX: the final classification_report now reflects the BEST checkpoint's
predictions, not whatever the last epoch trained happened to produce. Early
stopping runs `patience` more epochs past the peak before stopping, so the
last epoch is not the best one -- this is the same bug already fixed in
train_severity_real_interim.py, now fixed here too. The "Best QWK" summary
number was already tracked correctly; only the detailed per-class report
was stale.

Usage:
    python scripts/train_baseline_classifier.py --config configs/config.yaml --model efficientnet
    python scripts/train_baseline_classifier.py --config configs/config.yaml --model vit
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from sklearn.metrics import classification_report, cohen_kappa_score
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.baseline_dataset import RawImageSeverityDataset
from models.baselines.classifiers import EfficientNetB0Classifier, ViTClassifier
from utils.early_stopping import EarlyStopping

MODEL_REGISTRY = {
    "efficientnet": (EfficientNetB0Classifier, 224),
    "vit": (ViTClassifier, 224),
}


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels, n = 0.0, [], [], 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc="train" if train else "val", leave=False):
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            n += images.size(0)

    return total_loss / n, all_preds, all_labels


def main(config_path: str, model_name: str, epochs_override: int = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    sev_cfg = cfg["severity"]
    class_names = sev_cfg["classes"]
    epochs = epochs_override or sev_cfg.get("baseline_epochs", 30)

    model_cls, img_size = MODEL_REGISTRY[model_name]

    csv_path = Path(cfg["paths"]["severity_dataset"]) / "severity_labels.csv"
    images_root = Path(cfg["paths"]["segmentation_dataset"]) / "images"

    train_ds = RawImageSeverityDataset(csv_path, images_root, split="train", img_size=img_size, train=True)
    val_ds = RawImageSeverityDataset(csv_path, images_root, split="val", img_size=img_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    model = model_cls(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    ckpt_dir = Path(cfg["paths"]["checkpoints"]) / "baselines"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_kappa = -1.0
    best_val_preds, best_val_labels = None, None
    early_stopper = EarlyStopping(patience=sev_cfg.get("early_stopping_patience", 15), mode="max")

    for epoch in range(epochs):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_preds, val_labels = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        kappa = cohen_kappa_score(val_labels, val_preds, weights="quadratic") if len(set(val_labels)) > 1 else 0.0
        print(f"[{model_name}] Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_qwk={kappa:.4f}")

        is_best = early_stopper.step(kappa)
        if is_best:
            best_kappa = kappa
            best_val_preds, best_val_labels = val_preds, val_labels
            torch.save(model.state_dict(), ckpt_dir / f"{model_name}_classifier_best.pt")
            print(f"  -> saved new best checkpoint (qwk={kappa:.4f})")

        if early_stopper.should_stop:
            print(f"  -> early stopping triggered (no improvement for {early_stopper.patience} epochs)")
            break

    # BUG FIX: report from the BEST checkpoint's predictions, not the last epoch trained
    print(f"\n[{model_name}] Final classification report (BEST checkpoint, validation set):")
    print(classification_report(best_val_labels, best_val_preds, labels=list(range(len(class_names))),
                                 target_names=class_names, zero_division=0))
    print(f"[{model_name}] Best QWK: {best_kappa:.4f}")
    return best_kappa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.model, args.epochs)