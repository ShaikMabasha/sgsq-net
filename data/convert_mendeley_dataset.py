"""
Converts the real Mendeley gingivitis dataset (Reale-Nosei et al., 2024) into
the folder/CSV structure the rest of this pipeline expects.

CONFIRMED LABEL SCHEMA:
    class 0  -> upper-jaw gingiva region box
    class 1  -> lower-jaw gingiva region box
    class 2  -> per-tooth box, MGI level 0 (healthy)
    class 3  -> per-tooth box, MGI level 1 (mild)
    class 4  -> per-tooth box, MGI level 2 (moderate)
    class 5  -> per-tooth box, MGI level 3 (high)
    class 6  -> per-tooth box, MGI level 4 (severe)

SEVERITY LABEL: defined by MEAN MGI across all labeled teeth, quantile-binned
into 5 groups, edges fit on TRAIN split only.

SEGMENTATION MASKS: coarse rectangle-filled masks from per-tooth boxes with
MGI >= 1 (inflamed), keeping Stage 1/2/3 all on the SAME patient population
(avoids cross-dataset domain-shift failure seen when chaining a
segmentation model trained on a different dataset's photos).

FILENAME CONVENTION (bug fix): segmentation images/masks are saved with a
"{split}_" prefix (e.g. "train_1.jpg"), matching what train_segmentation.py's
glob-by-prefix logic expects. Detection files keep their ORIGINAL names,
organized into per-split folders instead (unaffected by this fix).

Usage:
    python data/convert_mendeley_dataset.py --source /path/to/Dataset --config configs/config.yaml
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

CLASS_TO_MGI = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
REGION_CLASSES = {0: "upper_gingiva", 1: "lower_gingiva"}

WORST_TOOTH_NAMES = [
    "healthy",
    "mild_gingivitis",
    "moderate_gingivitis",
    "high_gingivitis",
    "severe_gingivitis",
]

QUANTILE_SEVERITY_NAMES = [
    "grade_1_lowest_inflammation",
    "grade_2_low_inflammation",
    "grade_3_moderate_inflammation",
    "grade_4_high_inflammation",
    "grade_5_highest_inflammation",
]

SPLIT_MAP = {"Training": "train", "Validation": "val", "Test": "test"}


def write_yolo_detection_label(out_path: Path, mgi_boxes, region_boxes):
    """
    Remaps to this pipeline's 2-class detection scheme:
        0 = tooth   (any per-tooth box, regardless of MGI level)
        1 = gingiva (either jaw-region box)
    """
    lines = []
    for _, box in mgi_boxes:
        cx, cy, w, h = box
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    for _, box in region_boxes:
        cx, cy, w, h = box
        lines.append(f"1 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    out_path.write_text("\n".join(lines))


def build_box_mask(mgi_boxes_raw, img_w: int, img_h: int) -> np.ndarray:
    """
    Builds a COARSE binary segmentation mask by filling the rectangle of
    every per-tooth box with MGI >= 1 (inflamed) as foreground (255).
    Healthy teeth (MGI == 0) are left as 0.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for cls_id, box in mgi_boxes_raw:
        mgi_level = CLASS_TO_MGI[cls_id]
        if mgi_level < 1:
            continue
        cx, cy, w, h = box
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def convert(source_root: str, cfg: dict):
    source_root = Path(source_root)
    det_root = Path(cfg["paths"]["detection_dataset"])
    seg_root = Path(cfg["paths"]["segmentation_dataset"])
    sev_root = Path(cfg["paths"]["severity_dataset"])

    for split in ("train", "val", "test"):
        (det_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (det_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (seg_root / "images").mkdir(parents=True, exist_ok=True)
    (seg_root / "masks").mkdir(parents=True, exist_ok=True)
    sev_root.mkdir(parents=True, exist_ok=True)

    severity_rows = []
    skipped = []

    for dataset_split, pipeline_split in SPLIT_MAP.items():
        images_dir = source_root / dataset_split / "Images"
        labels_dir = source_root / dataset_split / "Labels"

        if not images_dir.exists():
            print(f"[skip] {images_dir} not found — is '{dataset_split}' the right folder name in your download?")
            continue

        image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
        print(f"{dataset_split}: found {len(image_paths)} images")

        for img_path in image_paths:
            stem = img_path.stem
            label_path = labels_dir / f"{stem}.txt"
            if not label_path.exists():
                skipped.append(f"{dataset_split}/{stem} (no matching label file)")
                continue

            mgi_boxes_raw = []
            region_boxes = []
            for line in label_path.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split()
                cls_id = int(parts[0])
                box = tuple(float(x) for x in parts[1:5])
                if cls_id in CLASS_TO_MGI:
                    mgi_boxes_raw.append((cls_id, box))
                elif cls_id in REGION_CLASSES:
                    region_boxes.append((cls_id, box))

            if len(mgi_boxes_raw) == 0:
                skipped.append(f"{dataset_split}/{stem} (no per-tooth boxes found)")
                continue

            mgi_levels = [CLASS_TO_MGI[c] for c, _ in mgi_boxes_raw]
            box_areas = [box[2] * box[3] for _, box in mgi_boxes_raw]

            image = cv2.imread(str(img_path))
            if image is None:
                skipped.append(f"{dataset_split}/{stem} (could not read image)")
                continue
            img_h, img_w = image.shape[:2]

            # Detection: keeps ORIGINAL filenames, organized by per-split folder
            dst_det_img = det_root / "images" / pipeline_split / img_path.name
            shutil.copy2(img_path, dst_det_img)

            # Segmentation: uses SPLIT-PREFIXED filenames (bug fix -- matches
            # train_segmentation.py's glob-by-prefix logic)
            out_stem = f"{pipeline_split}_{stem}"

            dst_seg_img = seg_root / "images" / f"{out_stem}.jpg"
            if not dst_seg_img.exists():
                shutil.copy2(img_path, dst_seg_img)

            write_yolo_detection_label(det_root / "labels" / pipeline_split / f"{stem}.txt",
                                        mgi_boxes_raw, region_boxes)

            box_mask = build_box_mask(mgi_boxes_raw, img_w, img_h)
            cv2.imwrite(str(seg_root / "masks" / f"{out_stem}.png"), box_mask)

            max_mgi = max(mgi_levels)
            mean_mgi = sum(mgi_levels) / len(mgi_levels)
            itf = sum(1 for m in mgi_levels if m >= 1) / len(mgi_levels)
            total_area = sum(box_areas)
            inflamed_area = sum(a for a, m in zip(box_areas, mgi_levels) if m >= 1)
            iaf = inflamed_area / total_area if total_area > 0 else 0.0

            severity_rows.append({
                "filename": f"{out_stem}.jpg",  # matches segmentation filenames (bug fix)
                "patient_id": stem,
                "split": pipeline_split,
                "severity_label_worst_tooth": max_mgi,
                "severity_name_worst_tooth": WORST_TOOTH_NAMES[max_mgi],
                "mean_mgi": round(mean_mgi, 4),
                "inflamed_tooth_fraction": round(itf, 4),
                "inflamed_area_fraction": round(iaf, 4),
                "num_teeth_labeled": len(mgi_levels),
            })

    df = pd.DataFrame(severity_rows)

    train_mean_mgi = df.loc[df["split"] == "train", "mean_mgi"].values
    if len(train_mean_mgi) < 5:
        raise ValueError("Not enough train samples to compute 5 quantile bins.")

    edges = np.quantile(train_mean_mgi, [0.2, 0.4, 0.6, 0.8])
    df["severity_label"] = np.digitize(df["mean_mgi"].values, edges).astype(int).clip(0, 4)
    df["severity_name"] = df["severity_label"].map(lambda i: QUANTILE_SEVERITY_NAMES[i])

    print(f"Quantile bin edges (mean_mgi, fit on train split only): {edges.tolist()}")

    df.to_csv(sev_root / "severity_labels.csv", index=False)

    data_yaml = det_root.parent / "detection_data.yaml"
    data_yaml.write_text(
        f"path: {det_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n  0: tooth\n  1: gingiva\n"
    )

    print(f"\nConverted {len(df)} images.")
    if skipped:
        print(f"Skipped {len(skipped)} images (see list below) — usually a missing label file:")
        for s in skipped[:20]:
            print(f"  {s}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    print(f"\nSeverity distribution — PRIMARY, quantile-binned mean-MGI (used for training):"
          f"\n{df['severity_name'].value_counts().sort_index()}")
    print(f"\nSeverity distribution — reference, worst-tooth MGI (NOT used for training):"
          f"\n{df['severity_name_worst_tooth'].value_counts()}")
    print(f"\nSplit sizes:\n{df['split'].value_counts()}")
    print(f"\nWrote severity_labels.csv to {sev_root / 'severity_labels.csv'}")
    print(f"Wrote detection data to {det_root}")
    print(f"Wrote box-derived coarse segmentation masks to {seg_root / 'masks'}")

    print("\n" + "=" * 70)
    print("REQUIRED CONFIG UPDATE — paste this into configs/config.yaml under 'severity:':")
    print("=" * 70)
    print("  classes:")
    for name in QUANTILE_SEVERITY_NAMES:
        print(f"    - \"{name}\"")

    return df


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    convert(args.source, cfg)
