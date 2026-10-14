"""
scripts/train_severity_groundtruth_continuous.py
====================================================
Trains a severity head on GROUND-TRUTH continuous ITF/IAF (see
scripts/compute_ground_truth_continuous_features.py) -- the same feature
FORMULA (mean per-tooth mgi_level / 4, area-weighted) as the detector-derived
head, but computed from true per-tooth annotations instead of detector
predictions.

This exists to make the "automation gap" comparison apples-to-apples: the
original oracle (severity_head_real_interim_best.pt) used a BINARY-threshold
ground-truth feature, while the new detector-derived head
(severity_head_detector_derived_best.pt) had to switch to a CONTINUOUS
feature because the binary version was degenerate on detector predictions.
Comparing those two checkpoints' QWKs conflates "detector accuracy" with
"feature formula quality." This script isolates the former: same continuous
formula on both sides, only the feature SOURCE (ground truth vs. detector)
differs.

Produces a THIRD checkpoint pair, separate from both existing severity
heads, so nothing gets overwritten:
    checkpoints/severity_head_gt_continuous_best.pt
    checkpoints/severity_feature_stats_gt_continuous.json

Usage:
    python3 scripts/train_severity_groundtruth_continuous.py --config configs/config.yaml
"""
import argparse
import sys
from pathlib import Path

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
        Path(cfg["paths"]["severity_dataset"]) / "severity_labels_groundtruth_continuous.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found -- run scripts/compute_ground_truth_continuous_features.py first."
        )

    df = pd.read_csv(csv_path)
    available_splits = set(df["split"].unique())
    val_split = "val" if "val" in available_splits else ("test" if "test" in available_splits else None)
    if val_split is None:
        raise ValueError(f"No val or test split found in {csv_path}. Available splits: {available_splits}")

    train_ds = RealBoxFeatureDataset(csv_path, split="train")
    val_ds = RealBoxFeatureDataset(csv_path, split=val_split)
    print(f"[gt-continuous features] Train: {len(train_ds)} samples | Val ({val_split}): {len(val_ds)} samples")

    train_features_raw = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    normalizer = FeatureNormalizer().fit(train_features_raw)
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    normalizer.save(ckpt_dir / "severity_feature_stats_gt_continuous.json")

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
        print(f"[gt-continuous] Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_qwk={kappa:.4f}")

        is_best = early_stopper.step(kappa)
        if is_best:
            best_kappa = kappa
            best_val_preds, best_val_labels = val_preds, val_labels
            torch.save(model.state_dict(), ckpt_dir / "severity_head_gt_continuous_best.pt")
            print(f"  -> saved new best checkpoint (qwk={kappa:.4f})")
        if early_stopper.should_stop:
            print("  -> early stopping triggered")
            break

    val_preds, val_labels = best_val_preds, best_val_labels
    print("\nFinal classification report (ground-truth continuous features, BEST checkpoint, validation set):")
    print(classification_report(val_labels, val_preds, labels=list(range(len(class_names))),
                                 target_names=class_names, zero_division=0))
    print(f"Best QWK (ground-truth continuous features): {best_kappa:.4f}")
    print("\n[reminder] This is a VALIDATION-split number. Run scripts/evaluate_test_set.py afterward "
          "for the held-out TEST-split apples-to-apples comparison against "
          "severity_head_detector_derived_best.pt (same feature formula, detector-derived source).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--csv-path", default=None,
                         help="Default: <severity_dataset>/severity_labels_groundtruth_continuous.csv")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.csv_path, args.epochs)
