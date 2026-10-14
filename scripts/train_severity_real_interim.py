"""
Interim real-data severity training — uses REAL, box-derived features
(`inflamed_tooth_fraction` and `inflamed_area_fraction`) computed from
ground-truth per-tooth labels, instead of IAR from a segmentation model's
predictions.

CRITICAL FIX: earlier versions of this script also fed `mean_mgi` as an
input feature. Since `severity_label` is a DETERMINISTIC function of
`mean_mgi` (np.digitize against fixed quantile edges in
convert_mendeley_dataset.py), including mean_mgi as a feature made the task
CIRCULAR — the model could trivially rediscover the exact thresholding rule
that created its own labels, rather than genuinely predicting severity from
observable evidence. This produced a spurious ~99% QWK that reflected
label leakage, not real predictive skill.

Fixed by training on TWO real, non-leaking features instead:
  - inflamed_tooth_fraction (ITF): plain count of inflamed teeth / total teeth
  - inflamed_area_fraction (IAF): same idea but AREA-weighted using each
    tooth's box size, giving continuous resolution instead of ITF's coarse
    ~13-step granularity
Neither feature is deterministic of severity_label, so this remains a
genuinely non-trivial prediction task.

WHY THIS SCRIPT EXISTS AT ALL: train_severity.py's default "predicted" mask
source needs a trained segmentation checkpoint. If that checkpoint was
trained on a DIFFERENT dataset's photos (e.g. Oral-Mamba) than the severity
labels come from (e.g. Mendeley), cross-dataset domain shift can make the
predicted IAR collapse to near-zero (observed: mean IAR ~0.0006). This
script sidesteps that entirely by using box-derived features computed
directly from Mendeley's own ground truth, with zero dependency on any
segmentation model's cross-dataset generalization.

Usage:
    python scripts/train_severity_real_interim.py --config configs/config.yaml
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


class RealBoxFeatureDataset(Dataset):
    """Reads inflamed_tooth_fraction + inflamed_area_fraction from severity_labels.csv.

    Both are box-derived, real, and — critically — NEITHER is deterministic
    of severity_label (unlike mean_mgi, which caused the label-leakage bug).
    """

    def __init__(self, csv_path: str, split: str):
        df = pd.read_csv(csv_path)
        required = {"inflamed_tooth_fraction", "inflamed_area_fraction", "severity_label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"severity_labels.csv is missing columns {missing} — "
                f"this script expects output from the UPDATED data/convert_mendeley_dataset.py "
                f"(with inflamed_area_fraction). Re-run the converter first."
            )
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise RuntimeError(f"No rows for split='{split}'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = torch.tensor(
            [row["inflamed_tooth_fraction"], row["inflamed_area_fraction"]], dtype=torch.float32
        )
        label = torch.tensor(int(row["severity_label"]), dtype=torch.long)
        return features, label


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels, n = 0.0, [], [], 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * features.size(0)
            preds = SeverityHead.logits_to_class(logits)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            n += features.size(0)
    return total_loss / n, all_preds, all_labels


def main(config_path: str, epochs_override: int = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    sev_cfg = cfg["severity"]
    class_names = sev_cfg["classes"]
    epochs = epochs_override or sev_cfg["epochs"]

    csv_path = Path(cfg["paths"]["severity_dataset"]) / "severity_labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found — run data/convert_mendeley_dataset.py first.")

    df = pd.read_csv(csv_path)
    available_splits = set(df["split"].unique())
    val_split = "val" if "val" in available_splits else ("test" if "test" in available_splits else None)
    if val_split is None:
        raise ValueError(f"No val or test split found in {csv_path}. Available splits: {available_splits}")

    train_ds = RealBoxFeatureDataset(csv_path, split="train")
    val_ds = RealBoxFeatureDataset(csv_path, split=val_split)
    print(f"Train: {len(train_ds)} samples | Val ({val_split}): {len(val_ds)} samples")

    train_features_raw = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    normalizer = FeatureNormalizer().fit(train_features_raw)
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    normalizer.save(ckpt_dir / "severity_feature_stats_real_interim.json")

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
        print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_qwk={kappa:.4f}")

        is_best = early_stopper.step(kappa)
        if is_best:
            best_kappa = kappa
            best_val_preds, best_val_labels = val_preds, val_labels
            torch.save(model.state_dict(), ckpt_dir / "severity_head_real_interim_best.pt")
            print(f"  -> saved new best checkpoint (qwk={kappa:.4f})")
        if early_stopper.should_stop:
            print(f"  -> early stopping triggered")
            break

    # BUG FIX: report metrics from the BEST checkpoint, not the last epoch trained
    val_preds, val_labels = best_val_preds, best_val_labels

    print("\nFinal classification report (real data, box-derived features, BEST checkpoint):")
    print(classification_report(val_labels, val_preds, labels=list(range(len(class_names))),
                                 target_names=class_names, zero_division=0))
    print(f"Best QWK (learned SeverityHead): {best_kappa:.4f}")

    # --- Diagnostic: naive rule-based thresholds, zero learned parameters ---
    train_labels_arr = train_ds.df["severity_label"].values
    val_labels_arr = val_ds.df["severity_label"].values
    learned_acc = (np.array(val_preds) == np.array(val_labels)).mean()

    print(f"\n[Rule-based diagnostics, 0 learned parameters each]")
    for feature_name in ("inflamed_tooth_fraction", "inflamed_area_fraction"):
        train_vals = train_ds.df[feature_name].values
        val_vals = val_ds.df[feature_name].values
        edges = np.quantile(train_vals, [0.2, 0.4, 0.6, 0.8])
        rule_preds = np.digitize(val_vals, edges).clip(0, len(class_names) - 1)
        rule_acc = (rule_preds == val_labels_arr).mean()
        rule_kappa = cohen_kappa_score(val_labels_arr, rule_preds, weights="quadratic") if len(set(val_labels_arr)) > 1 else 0.0
        print(f"  [{feature_name} threshold only]  Accuracy: {rule_acc:.4f} | QWK: {rule_kappa:.4f}")

    print(f"  [Learned SeverityHead, both features]  Accuracy: {learned_acc:.4f} | QWK: {best_kappa:.4f}")
    print(f"  Gap between learned and best single-feature rule tells you whether")
    print(f"  the learned mapping + feature combination adds real value.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.epochs)
