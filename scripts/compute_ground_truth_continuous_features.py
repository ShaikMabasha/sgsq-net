"""
scripts/compute_ground_truth_continuous_features.py
=======================================================
Closes a real apples-to-apples gap in the automation-gap comparison.

The "oracle" severity head (severity_head_real_interim_best.pt, QWK=0.259 on
TEST) was trained on ground-truth ITF/IAF computed with a BINARY threshold
(fraction of teeth with true mgi_level >= 1). The new detector-derived head
(severity_head_detector_derived_best.pt, trained on v2 features, QWK=0.467 on
TEST) uses a CONTINUOUS ordinal feature (mean predicted mgi_level / 4)
instead, because the binary version was completely degenerate on detector
predictions (see diagnose_detector_features.py's earlier finding: detector
never emits class 2 confidently, so binary ITF collapsed to a constant 1.0).

Comparing these two checkpoints' QWKs directly is NOT a clean "ground truth
vs. automated" comparison anymore -- part of the detector-derived head's gain
could be coming from the better (continuous) feature *formula*, not from the
detector's predictions being good. A reviewer could reasonably ask: "is your
detector-derived model beating the oracle because detection is accurate, or
just because continuous features are inherently better than binary ones?"

This script answers that by computing the SAME continuous formula
(mean per-tooth mgi_level / 4, and its area-weighted version) from
GROUND-TRUTH per-tooth boxes instead of detector predictions -- using the
label files already written by data/convert_mendeley_dataset_mgi_detection.py
at sgsq_data/detection_mgi/labels/{train,val,test}/<stem>.txt (same image
filenames as severity_labels.csv, same 7-class scheme, unmodified raw
annotations, no detector or GPU involved at all).

Output: sgsq_data/severity/severity_labels_groundtruth_continuous.csv
Train a severity head on it with:
    python3 scripts/train_severity_groundtruth_continuous.py --config configs/config.yaml
Then evaluate_test_set.py compares THREE severity heads on the same feature
formula: ground-truth-continuous (true apples-to-apples oracle) vs.
detector-derived-continuous (fully automated) vs. the original binary-oracle
(kept for continuity with earlier results).

Usage:
    python3 scripts/compute_ground_truth_continuous_features.py --config configs/config.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TOOTH_CLASSES = {2, 3, 4, 5, 6}
MGI_OFFSET = 2
MGI_MAX = 4.0


def compute_features_from_label_file(label_path: Path):
    """Mirrors compute_detector_derived_features_v2.py's formula exactly,
    but reads ground-truth YOLO-format label lines (no confidence column)
    instead of detector predictions."""
    if not label_path.exists():
        return 0.0, 0.0, 0

    lines = [l for l in label_path.read_text().strip().split("\n") if l.strip()]
    if not lines:
        return 0.0, 0.0, 0

    cls_list, area_list = [], []
    for line in lines:
        parts = line.split()
        cls_id = int(parts[0])
        if cls_id not in TOOTH_CLASSES:
            continue
        w, h = float(parts[3]), float(parts[4])
        cls_list.append(cls_id)
        area_list.append(w * h)

    if not cls_list:
        return 0.0, 0.0, 0

    mgi_levels = (np.array(cls_list) - MGI_OFFSET).astype(float)
    areas = np.array(area_list)

    mean_mgi_frac = float(mgi_levels.mean() / MGI_MAX)
    total_area = areas.sum()
    area_weighted_mgi = float((areas * mgi_levels).sum() / total_area) if total_area > 0 else 0.0
    area_weighted_mgi_frac = area_weighted_mgi / MGI_MAX

    return mean_mgi_frac, area_weighted_mgi_frac, len(cls_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--mgi-labels-dir", default=None,
                     help="Default: <detection_dataset>_mgi/labels (from "
                          "data/convert_mendeley_dataset_mgi_detection.py's output)")
    ap.add_argument("--out-csv", default=None,
                     help="Default: <severity_dataset>/severity_labels_groundtruth_continuous.csv")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sev_root = Path(cfg["paths"]["severity_dataset"])
    csv_path = sev_root / "severity_labels.csv"

    if args.mgi_labels_dir:
        mgi_labels_dir = Path(args.mgi_labels_dir)
    else:
        det_base = Path(cfg["paths"]["detection_dataset"])
        mgi_root = det_base.parent / f"{det_base.name}_mgi"
        mgi_labels_dir = mgi_root / "labels"

    if not mgi_labels_dir.exists():
        raise FileNotFoundError(
            f"{mgi_labels_dir} not found -- run data/convert_mendeley_dataset_mgi_detection.py first."
        )

    out_csv = Path(args.out_csv) if args.out_csv else sev_root / "severity_labels_groundtruth_continuous.csv"

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}. Computing ground-truth continuous "
          f"ITF/IAF from {mgi_labels_dir} (no GPU, no detector -- pure ground-truth labels)...")

    itfs, iafs, n_teeth_list = [], [], []
    n_missing = 0
    for i, row in df.iterrows():
        stem = Path(row["filename"]).stem
        # severity_labels.csv names images "<split>_00001.jpg" (e.g. "train_00001"),
        # but data/convert_mendeley_dataset_mgi_detection.py preserves the raw source
        # stem with no split prefix ("00001.txt") under labels/<split>/. Strip the
        # "<split>_" prefix this pipeline adds before looking up the MGI label file.
        split_prefix = f"{row['split']}_"
        bare_stem = stem[len(split_prefix):] if stem.startswith(split_prefix) else stem
        label_path = mgi_labels_dir / row["split"] / f"{bare_stem}.txt"
        if not label_path.exists():
            n_missing += 1
            itfs.append(0.0)
            iafs.append(0.0)
            n_teeth_list.append(0)
            continue
        itf, iaf, n_teeth = compute_features_from_label_file(label_path)
        itfs.append(itf)
        iafs.append(iaf)
        n_teeth_list.append(n_teeth)

    df_out = df.copy()
    df_out["inflamed_tooth_fraction"] = itfs
    df_out["inflamed_area_fraction"] = iafs
    df_out["n_teeth_detected"] = n_teeth_list
    df_out.to_csv(out_csv, index=False)

    itf_arr = np.array(itfs)
    iaf_arr = np.array(iafs)
    print(f"\nDone. Wrote {out_csv}")
    print(f"Missing label files: {n_missing} / {len(df)}")
    print(f"inflamed_tooth_fraction (ground-truth continuous): mean={itf_arr.mean():.4f} "
          f"std={itf_arr.std():.4f} min={itf_arr.min():.4f} max={itf_arr.max():.4f}")
    print(f"inflamed_area_fraction  (ground-truth continuous): mean={iaf_arr.mean():.4f} "
          f"std={iaf_arr.std():.4f} min={iaf_arr.min():.4f} max={iaf_arr.max():.4f}")

    print(f"\nPer-severity_label breakdown:")
    print(df_out.groupby("severity_label")[["inflamed_tooth_fraction", "inflamed_area_fraction"]].describe().T)

    print("\nNext: python3 scripts/train_severity_groundtruth_continuous.py --config configs/config.yaml")


if __name__ == "__main__":
    main()
