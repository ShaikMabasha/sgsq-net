"""
scripts/train_severity_detector_derived.py
=============================================
Trains the severity head on DETECTOR-DERIVED box features (ITF/IAF computed
from the 7-class MGI detector's own predictions -- see
scripts/compute_detector_derived_features.py), rather than the ground-truth
per-tooth annotations used by scripts/train_severity_real_interim.py.

WHY A SEPARATE TRAINING RUN, NOT JUST A DIFFERENT EVAL: your own project
already established the right principle for this (see
data/dataset.py's SeverityFeatureDataset docstring on mask_source="predicted"
vs "ground_truth" -- "Stage 3 must learn on the same kind of noisy masks it
will receive at real inference time, not the clean ground truth it will
never see again after training"). The same logic applies here: evaluating a
head trained on clean ground-truth ITF/IAF against noisy detector-derived
ITF/IAF at test time would UNDERSTATE what's achievable, because the model
never learned to be robust to detector noise. Training directly on
detector-derived features (as this script does) gives the fair, "what a
deployed system would actually achieve" number.

This produces a SECOND checkpoint (severity_head_detector_derived_best.pt)
alongside your existing severity_head_real_interim_best.pt (trained on
ground-truth features). Evaluating both on the held-out test set and
comparing them (scripts/evaluate_test_set.py does this automatically once
both exist) tells you the "automation gap" -- how much QWK you lose by using
a real, fully-automated detector instead of an oracle with ground-truth
per-tooth grades.

Usage:
    python3 scripts/train_severity_detector_derived.py --config configs/config.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import yaml
from sklearn.metrics import classification_report, cohen_kappa_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.severity_head import SeverityHead, CoralLoss, FeatureNormalizer
from utils.early_stopping import EarlyStopping
from scripts.train_severity_real_interim import RealBoxFeatureDataset, run_epoch


def main(config_path: str, csv_path_override: str, epochs_override: int = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    sev_cfg = cfg["severity"]
    class_names = sev_cfg["classes"]
    epochs = epochs_override or sev_cfg["epochs"]

    csv_path = Path(csv_path_override) if csv_path_override else \
        Path(cfg["paths"]["severity_dataset"]) / "severity_labels_detector_derived.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found -- run scripts/compute_detector_derived_features.py first "
            f"(which itself needs scripts/train_detection_mgi.py's checkpoint)."
        )

    df = pd.read_csv(csv_path)
    available_splits = set(df["split"].unique())
    val_split = "val" if "val" in available_splits else ("test" if "test" in available_splits else None)
    if val_split is None:
        raise ValueError(f"No val or test split found in {csv_path}. Available splits: {available_splits}")

    train_ds = RealBoxFeatureDataset(csv_path, split="train")
    val_ds = RealBoxFeatureDataset(csv_path, split=val_split)
    print(f"[detector-derived features] Train: {len(train_ds)} samples | Val ({val_split}): {len(val_ds)} samples")

    train_features_raw = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    normalizer = FeatureNormalizer().fit(train_features_raw)
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    normalizer.save(ckpt_dir / "severity_feature_stats_detector_derived.json")

    class NormWrapper(Dataset):
        def __init__(self, base, norm):
            self.base, self.norm = base, norm
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            f, l = self.base[i]
            return self.norm.transform(f.unsqueeze(0)).squeeze(0), l

    train_loader = DataLoader(NormWrapper(train_ds, normalizer), batch_size=sev_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(NormWrapper(val_ds, normalizer), batch_size=sev_cfg["batch_size"], shuffle=False)

    model = SeverityHead(num_classes=len(class_names), input_dim=2).to(device)
    criterion = CoralLoss(num_classes=len(class_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=sev_cfg["lr"])
    early_stopper = EarlyStopping(patience=sev_cfg.get("early_stopping_patience", 15), mode="max")

    best_kappa = -1.0
    best_val_preds, best_val_labels = None, None
    for epoch in range(epochs):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_preds, val_labels = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        kappa = cohen_kappa_score(val_labels, val_preds, weights="quadratic") if len(set(val_labels)) > 1 else 0.0
        print(f"[detector-derived] Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_qwk={kappa:.4f}")

        is_best = early_stopper.step(kappa)
        if is_best:
            best_kappa = kappa
            best_val_preds, best_val_labels = val_preds, val_labels
            torch.save(model.state_dict(), ckpt_dir / "severity_head_detector_derived_best.pt")
            print(f"  -> saved new best checkpoint (qwk={kappa:.4f})")
        if early_stopper.should_stop:
            print("  -> early stopping triggered")
            break

    val_preds, val_labels = best_val_preds, best_val_labels
    print("\nFinal classification report (detector-derived features, BEST checkpoint, validation set):")
    print(classification_report(val_labels, val_preds, labels=list(range(len(class_names))),
                                 target_names=class_names, zero_division=0))
    print(f"Best QWK (detector-derived features): {best_kappa:.4f}")
    print("\n[reminder] This is a VALIDATION-split number, for early-stopping/sanity-check purposes only. "
          "Run scripts/evaluate_test_set.py afterward for the held-out TEST-split comparison against "
          "the ground-truth-feature oracle (severity_head_real_interim_best.pt).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--csv-path", default=None,
                         help="Default: <severity_dataset>/severity_labels_detector_derived.csv")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.csv_path, args.epochs)
