"""
scripts/diagnose_detector_features.py
=======================================
Quick, cheap (no GPU, no model load) diagnostic for why the detector-derived
severity head collapsed to a single constant class. Inspects the actual
feature distribution in severity_labels_detector_derived.csv, split by the
TRUE severity_label, to check whether inflamed_tooth_fraction / IAF have
enough variance to be discriminative at all.

Usage:
    python3 scripts/diagnose_detector_features.py --config configs/config.yaml
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--csv", default=None,
                     help="Default: <severity_dataset>/severity_labels_detector_derived.csv")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sev_root = Path(cfg["paths"]["severity_dataset"])
    csv_path = Path(args.csv) if args.csv else sev_root / "severity_labels_detector_derived.csv"

    df = pd.read_csv(csv_path)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    print(f"Loaded {len(df)} rows from {csv_path}\n")

    print("=" * 70)
    print("Overall feature distribution (all splits)")
    print("=" * 70)
    print(df[["inflamed_tooth_fraction", "inflamed_area_fraction", "n_teeth_detected"]].describe())

    print("\n" + "=" * 70)
    print("Feature distribution BY TRUE severity_label (this is the key check --")
    print("if ITF/IAF medians and IQRs barely differ across severity_label groups,")
    print("the detector-derived features carry ~no discriminative signal)")
    print("=" * 70)
    grp = df.groupby("severity_label")[["inflamed_tooth_fraction", "inflamed_area_fraction"]]
    print(grp.describe().T)

    print("\n" + "=" * 70)
    print("Value counts: how often does ITF land exactly at 0.0 or 1.0?")
    print("(near-universal saturation at one end = no signal left)")
    print("=" * 70)
    print("ITF == 0.0:", (df["inflamed_tooth_fraction"] == 0.0).mean())
    print("ITF == 1.0:", (df["inflamed_tooth_fraction"] == 1.0).mean())
    print("ITF std dev:", df["inflamed_tooth_fraction"].std())
    print("IAF std dev:", df["inflamed_area_fraction"].std())

    print("\n" + "=" * 70)
    print("Compare to GROUND-TRUTH ITF/IAF from the original severity_labels.csv,")
    print("if present alongside, to see how much variance the detector destroyed")
    print("=" * 70)
    gt_csv = sev_root / "severity_labels.csv"
    if gt_csv.exists():
        gt = pd.read_csv(gt_csv)
        common_cols = [c for c in ["inflamed_tooth_fraction", "inflamed_area_fraction"] if c in gt.columns]
        if common_cols:
            print("Ground-truth (oracle) feature std dev:")
            print(gt[common_cols].std())
        else:
            print(f"  (ground-truth csv found but has no {['inflamed_tooth_fraction','inflamed_area_fraction']} columns -- skipping)")
    else:
        print(f"  ({gt_csv} not found -- skipping comparison)")

    print("\nDone. If per-severity_label ITF/IAF distributions overlap heavily and "
          "std dev is much smaller than the ground-truth std dev, that confirms the "
          "detector-derived features are the bottleneck, not the severity head itself.")


if __name__ == "__main__":
    main()
