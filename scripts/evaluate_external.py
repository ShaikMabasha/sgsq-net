"""
Evaluates trained models on the external/OOD dataset
(data/generate_external_data.py) — the "external validation" result.

Usage:
    python scripts/evaluate_external.py --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score, f1_score
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.dataset import SegmentationDataset
from data.dataset import SeverityFeatureDataset
from models.unet import ResNetUNet
from models.severity_head import SeverityHead, FeatureNormalizer
from scripts.train_segmentation import dice_score


def evaluate_segmentation_external(cfg: dict, ext_seg_root: str):
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    seg_cfg = cfg["segmentation"]
    ckpt_path = Path(cfg["paths"]["checkpoints"]) / "segmentation_best.pt"

    if not ckpt_path.exists():
        print(f"[skip] no segmentation checkpoint at {ckpt_path}")
        return None

    model = ResNetUNet(num_classes=seg_cfg["num_classes"]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    filenames = sorted([p.name for p in (Path(ext_seg_root) / "images").glob("*.jpg")])
    ds = SegmentationDataset(ext_seg_root, filenames=filenames, img_size=seg_cfg["img_size"], train=False)
    loader = DataLoader(ds, batch_size=seg_cfg["batch_size"], shuffle=False)

    dices = []
    with torch.no_grad():
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            dices.append(dice_score(logits, masks) * images.size(0))

    mean_dice = sum(dices) / len(ds)
    print(f"[External validation] Segmentation Dice: {mean_dice:.4f} (n={len(ds)})")
    return mean_dice


def evaluate_severity_external(cfg: dict, ext_csv_path: str, ext_seg_root: str):
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    seg_cfg = cfg["segmentation"]
    sev_cfg = cfg["severity"]
    class_names = sev_cfg["classes"]
    ckpt_dir = Path(cfg["paths"]["checkpoints"])

    seg_ckpt = ckpt_dir / "segmentation_best.pt"
    sev_ckpt = ckpt_dir / "severity_head_best.pt"
    norm_stats = ckpt_dir / "severity_feature_stats.json"

    if not (seg_ckpt.exists() and sev_ckpt.exists() and norm_stats.exists()):
        print("[skip] missing one or more required checkpoints for severity external evaluation")
        return None

    seg_model = ResNetUNet(num_classes=seg_cfg["num_classes"]).to(device)
    seg_model.load_state_dict(torch.load(seg_ckpt, map_location=device))
    seg_model.eval()

    sev_model = SeverityHead(num_classes=len(class_names), input_dim=2).to(device)
    sev_model.load_state_dict(torch.load(sev_ckpt, map_location=device))
    sev_model.eval()

    normalizer = FeatureNormalizer().load(norm_stats)

    df = pd.read_csv(ext_csv_path)
    ds = SeverityFeatureDataset.__new__(SeverityFeatureDataset)
    ds.df = df.reset_index(drop=True)
    ds.images_dir = Path(ext_seg_root) / "images"
    ds.masks_dir = Path(ext_seg_root) / "masks"
    ds.mask_source = "predicted"
    ds._features = ds._compute_predicted_features(seg_model, device, seg_cfg["img_size"])

    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            features_raw, label = ds[i]
            features = normalizer.transform(features_raw.unsqueeze(0).to(device))
            logits = sev_model(features)
            pred = SeverityHead.logits_to_class(logits).item()
            all_preds.append(pred)
            all_labels.append(label.item())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic") if len(set(all_labels)) > 1 else 0.0

    print(f"\n[External validation] Severity — Accuracy: {acc:.4f} | Macro-F1: {f1:.4f} | QWK: {kappa:.4f}")
    print(classification_report(all_labels, all_preds, labels=list(range(len(class_names))),
                                 target_names=class_names, zero_division=0))
    return {"accuracy": acc, "macro_f1": f1, "qwk": kappa}


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ext_root = Path(cfg.get("external_validation", {}).get("root", "sgsq_data_external"))
    ext_seg_root = ext_root / "segmentation"
    ext_csv = ext_root / "severity_labels_external.csv"

    if not ext_csv.exists():
        raise FileNotFoundError(
            f"{ext_csv} not found — run data/generate_external_data.py first."
        )

    print("=" * 60)
    print("EXTERNAL / OUT-OF-DISTRIBUTION VALIDATION")
    print("=" * 60)
    seg_dice = evaluate_segmentation_external(cfg, ext_seg_root)
    sev_results = evaluate_severity_external(cfg, ext_csv, ext_seg_root)

    results = {"external_segmentation_dice": seg_dice}
    if sev_results:
        results.update({f"external_severity_{k}": v for k, v in sev_results.items()})

    out_path = Path(cfg["paths"]["checkpoints"]) / "external_validation_results.csv"
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
