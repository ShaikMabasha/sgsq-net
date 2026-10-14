"""
scripts/compute_detector_derived_features_v2.py
==================================================
Fixes a real bug found in compute_detector_derived_features.py: that script
computed ITF as a BINARY threshold (fraction of detected teeth with
mgi_level >= 1). Since the 7-class detector almost never emits class 2
("tooth_mgi0_healthy") above the confidence threshold (confirmed via
diagnose_detector_features.py: ITF/IAF were EXACTLY 1.0 for all 1096 images,
zero variance), that binary feature is completely degenerate -- even though
the detector DOES distinguish severity reasonably among the classes it can
predict (mAP50-95: mild=0.098, moderate=0.263, high=0.224, severe=0.366).

This version replaces the binary threshold with a CONTINUOUS ordinal
feature: the mean predicted MGI level (0-4 scale, normalized to [0,1] by
dividing by 4) across detected teeth, and its area-weighted counterpart.
This keeps the graded severity information the detector actually captures,
instead of collapsing it through a >=1 threshold. Column names are kept
identical to the original script's output (inflamed_tooth_fraction,
inflamed_area_fraction) so it's a drop-in replacement for every downstream
script (RealBoxFeatureDataset, train_severity_detector_derived.py,
evaluate_test_set.py) -- no other code changes required.

Usage:
    python3 scripts/compute_detector_derived_features_v2.py --config configs/config.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

TOOTH_CLASSES = {2, 3, 4, 5, 6}  # per-tooth MGI classes in the 7-class scheme
MGI_OFFSET = 2       # mgi_level = predicted_class - MGI_OFFSET  (range 0-4)
MGI_MAX = 4.0         # for normalizing mean mgi_level to [0, 1]


def compute_features_for_image(result, conf_threshold: float):
    """
    Returns (mean_mgi_frac, area_weighted_mgi_frac, n_teeth_detected).
    Both fractions are the mean/area-weighted-mean predicted MGI level
    (0-4), normalized by MGI_MAX to land in [0, 1] -- same scale/column
    names as the old binary ITF/IAF, but continuous instead of thresholded.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 0.0, 0.0, 0

    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    xywhn = boxes.xywhn.cpu().numpy()  # normalized [cx, cy, w, h]

    keep = (conf >= conf_threshold) & pd.Series(cls).isin(TOOTH_CLASSES).values
    cls = cls[keep]
    xywhn = xywhn[keep]

    if len(cls) == 0:
        return 0.0, 0.0, 0

    mgi_levels = (cls - MGI_OFFSET).astype(float)  # 0..4
    areas = xywhn[:, 2] * xywhn[:, 3]

    mean_mgi_frac = float(mgi_levels.mean() / MGI_MAX)
    total_area = areas.sum()
    area_weighted_mgi = float((areas * mgi_levels).sum() / total_area) if total_area > 0 else 0.0
    area_weighted_mgi_frac = area_weighted_mgi / MGI_MAX

    return mean_mgi_frac, area_weighted_mgi_frac, int(len(cls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--detector-ckpt", default=None,
                     help="Default: <checkpoints>/detection_mgi_best.pt")
    ap.add_argument("--conf-threshold", type=float, default=0.25)
    ap.add_argument("--out-csv", default=None,
                     help="Default: <severity_dataset>/severity_labels_detector_derived_v2.csv")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    seg_root = Path(cfg["paths"]["segmentation_dataset"])
    sev_root = Path(cfg["paths"]["severity_dataset"])
    csv_path = sev_root / "severity_labels.csv"

    detector_ckpt = Path(args.detector_ckpt) if args.detector_ckpt else ckpt_dir / "detection_mgi_best.pt"
    if not detector_ckpt.exists():
        raise FileNotFoundError(
            f"{detector_ckpt} not found -- train it first with scripts/train_detection_mgi.py, "
            f"or pass --detector-ckpt explicitly."
        )
    out_csv = Path(args.out_csv) if args.out_csv else sev_root / "severity_labels_detector_derived_v2.csv"

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path} (all splits). Running detector inference "
          f"(continuous ordinal MGI features this time, not binary threshold)...")

    model = YOLO(str(detector_ckpt))

    itfs, iafs, n_teeth_list = [], [], []
    n_zero_detections = 0
    images_dir = seg_root / "images"

    for i, row in df.iterrows():
        img_path = images_dir / row["filename"]
        if not img_path.exists():
            raise FileNotFoundError(f"{img_path} not found (from severity_labels.csv row {i})")
        results = model.predict(str(img_path), verbose=False)
        itf, iaf, n_teeth = compute_features_for_image(results[0], args.conf_threshold)
        if n_teeth == 0:
            n_zero_detections += 1
        itfs.append(itf)
        iafs.append(iaf)
        n_teeth_list.append(n_teeth)
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(df)} images processed")

    df_out = df.copy()
    df_out["inflamed_tooth_fraction"] = itfs   # now: mean predicted MGI level / 4 (continuous)
    df_out["inflamed_area_fraction"] = iafs    # now: area-weighted mean predicted MGI level / 4
    df_out["n_teeth_detected"] = n_teeth_list
    df_out.to_csv(out_csv, index=False)

    print(f"\nDone. Wrote {out_csv}")
    print(f"Images with ZERO teeth detected: {n_zero_detections} / {len(df)} "
          f"({100 * n_zero_detections / len(df):.1f}%)")

    itf_arr = np.array(itfs)
    iaf_arr = np.array(iafs)
    print(f"\nNew feature variance check (should be > 0 now):")
    print(f"  inflamed_tooth_fraction (continuous): mean={itf_arr.mean():.4f} std={itf_arr.std():.4f} "
          f"min={itf_arr.min():.4f} max={itf_arr.max():.4f}")
    print(f"  inflamed_area_fraction  (continuous): mean={iaf_arr.mean():.4f} std={iaf_arr.std():.4f} "
          f"min={iaf_arr.min():.4f} max={iaf_arr.max():.4f}")
    if itf_arr.std() < 1e-6 and iaf_arr.std() < 1e-6:
        print("  [warning] STILL zero variance -- the continuous fix did not recover signal. "
              "This points to the detector itself, not the feature formula -- see the "
              "class-balanced retrain option next.")

    print(f"\nPer-split row counts:\n{df_out['split'].value_counts()}")
    print("\nNext: python3 scripts/train_severity_detector_derived.py --config configs/config.yaml "
          f"--csv-path {out_csv}")
    print("(This will OVERWRITE checkpoints/severity_head_detector_derived_best.pt and "
          "severity_feature_stats_detector_derived.json -- fine, since the old binary-threshold "
          "version was a degenerate QWK=0.0000 checkpoint anyway. evaluate_test_set.py already "
          "points at these same filenames, so no other changes needed to re-evaluate on TEST.)")


if __name__ == "__main__":
    main()
