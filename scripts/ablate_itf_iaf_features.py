"""
scripts/ablate_itf_iaf_features.py
====================================
Feature-ablation experiment: trains three otherwise-identical severity heads
that differ only in which detector-derived feature(s) they see --

    ITF only   (inflamed_tooth_fraction          -- mean predicted MGI level / 4)
    IAF only   (inflamed_area_fraction            -- area-weighted mean predicted MGI level / 4)
    ITF + IAF  (both -- this reproduces the existing primary checkpoint's setup,
                retrained here with the SAME code path so the three numbers are
                directly comparable, rather than diffing this script's QWK against
                a checkpoint trained by a different script at a different time)

This answers the reviewer-relevant question "which feature is actually doing
the work?" -- cheap to run (the severity head is a tiny feed-forward network;
each config trains in well under a minute on a GPU, the slow part is just
loading the CSV once).

Does NOT touch any existing checkpoint. Writes three new, separate files:
    checkpoints/severity_head_ablation_itf_only_best.pt
    checkpoints/severity_head_ablation_iaf_only_best.pt
    checkpoints/severity_head_ablation_itf_iaf_best.pt
    checkpoints/severity_feature_stats_ablation_<mode>.json   (one per mode)
    checkpoints/ablation_itf_iaf_report.json                 (final summary, all 3)

Usage (from the sgsq_pipeline root, same place you run evaluate_test_set.py):

    python3 scripts/ablate_itf_iaf_features.py --config configs/config.yaml \
        --csv-path sgsq_data/severity/severity_labels_detector_derived_v2.csv

Pass whichever detector-derived CSV you've been using for the primary result
(the one evaluate_test_set.py's --detector-derived-csv points at). If you
omit --csv-path, it defaults to <severity_dataset>/severity_labels_detector_derived_v2.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import yaml
from sklearn.metrics import classification_report, cohen_kappa_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.severity_head import SeverityHead, CoralLoss, FeatureNormalizer
from utils.early_stopping import EarlyStopping
from stats_utils import compute_qwk_bootstrap

FEATURE_COLUMNS = {
    "itf_only": ["inflamed_tooth_fraction"],
    "iaf_only": ["inflamed_area_fraction"],
    "itf_iaf": ["inflamed_tooth_fraction", "inflamed_area_fraction"],
}


class FeatureSubsetDataset(Dataset):
    """Reads severity_label + a chosen subset of {inflamed_tooth_fraction,
    inflamed_area_fraction} straight from the detector-derived CSV. Written
    standalone (not reusing RealBoxFeatureDataset) so the feature subset is
    just a column-list argument, with no change to any existing class.
    """
    def __init__(self, csv_path, split: str, columns: list):
        df = pd.read_csv(csv_path)
        if "split" not in df.columns:
            raise ValueError(f"{csv_path} has no 'split' column -- is this the right CSV?")
        df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows with split=='{split}' in {csv_path}")
        missing = [c for c in columns + ["severity_label"] if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing column(s) {missing}. "
                              f"Available columns: {list(df.columns)}")
        self.features = df[columns].values.astype("float32")
        self.labels = df["severity_label"].values.astype("int64")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.tensor(self.features[i]), torch.tensor(self.labels[i])


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * feats.size(0)
            all_preds.extend(SeverityHead.logits_to_class(logits).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return total_loss / len(loader.dataset), all_preds, all_labels


def train_one_config(mode: str, cfg, csv_path: Path, ckpt_dir: Path, device, epochs: int, n_boot: int) -> dict:
    columns = FEATURE_COLUMNS[mode]
    sev_cfg = cfg["severity"]
    class_names = sev_cfg["classes"]
    num_classes = len(class_names)

    train_ds = FeatureSubsetDataset(csv_path, "train", columns)
    val_ds = FeatureSubsetDataset(csv_path, "val", columns)
    test_ds = FeatureSubsetDataset(csv_path, "test", columns)
    print(f"\n[{mode}] features={columns} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_features_raw = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    normalizer = FeatureNormalizer().fit(train_features_raw)
    normalizer.save(ckpt_dir / f"severity_feature_stats_ablation_{mode}.json")

    class NormWrapper(Dataset):
        def __init__(self, base, norm):
            self.base, self.norm = base, norm
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            f, l = self.base[i]
            return self.norm.transform(f.unsqueeze(0)).squeeze(0), l

    batch_size = sev_cfg["batch_size"]
    train_loader = DataLoader(NormWrapper(train_ds, normalizer), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NormWrapper(val_ds, normalizer), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(NormWrapper(test_ds, normalizer), batch_size=batch_size, shuffle=False)

    model = SeverityHead(num_classes=num_classes, input_dim=len(columns)).to(device)
    criterion = CoralLoss(num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=sev_cfg["lr"])
    early_stopper = EarlyStopping(patience=sev_cfg.get("early_stopping_patience", 15), mode="max")

    ckpt_path = ckpt_dir / f"severity_head_ablation_{mode}_best.pt"
    best_kappa = -1.0
    for epoch in range(epochs):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_preds, val_labels = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        kappa = cohen_kappa_score(val_labels, val_preds, weights="quadratic") if len(set(val_labels)) > 1 else 0.0
        print(f"[{mode}] epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_qwk={kappa:.4f}")

        is_best = early_stopper.step(kappa)
        if is_best:
            best_kappa = kappa
            torch.save(model.state_dict(), ckpt_path)
        if early_stopper.should_stop:
            print(f"[{mode}] early stopping triggered")
            break

    # Reload the BEST checkpoint (not whatever the last epoch happened to be)
    # before touching the held-out TEST split -- same reporting-from-last-epoch
    # bug this project already hit and fixed elsewhere; avoided here by never
    # reading val_preds/val_labels directly as the "final" numbers.
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for feats, labels in test_loader:
            feats = feats.to(device)
            logits = model(feats)
            test_preds.extend(SeverityHead.logits_to_class(logits).cpu().tolist())
            test_labels.extend(labels.tolist())
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)

    qwk_ci = compute_qwk_bootstrap(test_labels, test_preds, num_classes, n_boot=n_boot)
    print(f"\n[{mode}] HELD-OUT TEST: QWK = {qwk_ci['point']:.4f}  "
          f"[95% CI {qwk_ci['ci_low']:.4f}, {qwk_ci['ci_high']:.4f}]  (n={qwk_ci['n']}, best_val_qwk={best_kappa:.4f})")
    print(classification_report(test_labels, test_preds, labels=list(range(num_classes)),
                                 target_names=class_names, zero_division=0))

    return {
        "mode": mode,
        "features": columns,
        "best_val_qwk": best_kappa,
        "test_qwk": qwk_ci,
        "test_preds": test_preds.tolist(),
        "test_labels": test_labels.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--csv-path", default=None,
                     help="Detector-derived CSV (the same one used for the primary result). "
                          "Default: <severity_dataset>/severity_labels_detector_derived_v2.csv")
    ap.add_argument("--epochs", type=int, default=None, help="Default: severity.epochs from config")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))
    print(f"Device: {device}")

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_path) if args.csv_path else \
        Path(cfg["paths"]["severity_dataset"]) / "severity_labels_detector_derived_v2.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run scripts/compute_detector_derived_features_v2.py first, "
            f"or pass --csv-path pointing at your detector-derived CSV."
        )

    epochs = args.epochs or cfg["severity"]["epochs"]

    results = {}
    for mode in ("itf_only", "iaf_only", "itf_iaf"):
        results[mode] = train_one_config(mode, cfg, csv_path, ckpt_dir, device, epochs, args.n_boot)

    print("\n" + "=" * 70)
    print("FEATURE ABLATION SUMMARY -- HELD-OUT TEST SPLIT")
    print("=" * 70)
    print(f"{'Feature config':<16} {'QWK':>8}   95% CI")
    for mode in ("itf_only", "iaf_only", "itf_iaf"):
        q = results[mode]["test_qwk"]
        print(f"{mode:<16} {q['point']:>8.4f}   [{q['ci_low']:.4f}, {q['ci_high']:.4f}]")

    out_path = ckpt_dir / "ablation_itf_iaf_report.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print("\nUse the ITF-only and IAF-only TEST QWKs above to say, in the paper, which signal the")
    print("detector-derived pathway's QWK=0.467 result actually comes from. The 'itf_iaf' row here")
    print("is a same-code-path retrain of the existing primary result for a clean comparison --")
    print("expect it close to, not necessarily identical to, the already-reported 0.467 (different")
    print("random init/training run), which is fine for ablation purposes.")


if __name__ == "__main__":
    main()
