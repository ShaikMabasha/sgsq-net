"""
data/convert_mendeley_dataset_mgi_detection.py
=================================================
Adds a SECOND detection label scheme to the Mendeley converter, without
touching your existing detection/segmentation/severity data.

WHY THIS EXISTS: the box-derived severity features (inflamed_tooth_fraction,
inflamed_area_fraction -- the ones that gave the real, non-leaked QWK=0.259
result on the held-out test set) were computed directly from the dataset's
GROUND-TRUTH per-tooth MGI labels. Your trained YOLO detector (from the
original `data/convert_mendeley_dataset.py`) was deliberately collapsed to a
2-class scheme (0=tooth, 1=gingiva) -- see that file's own
`write_yolo_detection_label` docstring: "Severity information is NOT encoded
here." That means the detector, as currently trained, structurally CANNOT
reproduce ITF/IAF at real inference time on a new photo: the per-tooth MGI
level it would need was never in its label space to begin with.

This script writes a SEPARATE 7-class detection dataset (2 jaw-region classes
+ 5 per-tooth MGI-level classes, i.e. the RAW label scheme straight from the
source data, unmodified) to its own folder, alongside your existing 2-class
one. Training a detector on this makes per-tooth MGI level something the
model actually predicts, which is what closes the gap and lets you compute
ITF/IAF from a detector's own predictions on unseen images -- see
`scripts/compute_detector_derived_features.py`, which consumes this
detector's checkpoint.

Nothing here touches sgsq_data/detection, sgsq_data/segmentation, or
sgsq_data/severity/severity_labels.csv -- those keep working exactly as
before. Only a new sgsq_data/detection_mgi/ folder + detection_mgi_data.yaml
are written.

Usage (from the pipeline root, same --source you used before):
    python3 data/convert_mendeley_dataset_mgi_detection.py \\
        --source /path/to/Dataset --config configs/config.yaml
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

# Raw source label scheme, unchanged -- this IS the 7-class scheme we want
# the detector to learn directly, so no remapping happens at all here
# (contrast with convert_mendeley_dataset.py's write_yolo_detection_label,
# which collapses these 7 into 2).
CLASS_NAMES_7 = {
    0: "upper_gingiva",
    1: "lower_gingiva",
    2: "tooth_mgi0_healthy",
    3: "tooth_mgi1_mild",
    4: "tooth_mgi2_moderate",
    5: "tooth_mgi3_high",
    6: "tooth_mgi4_severe",
}
VALID_CLASSES = set(CLASS_NAMES_7.keys())

SPLIT_MAP = {"Training": "train", "Validation": "val", "Test": "test"}


def convert(source_root: str, det_root: Path):
    source_root = Path(source_root)
    for split in ("train", "val", "test"):
        (det_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (det_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_written, n_skipped = 0, 0

    for dataset_split, pipeline_split in SPLIT_MAP.items():
        images_dir = source_root / dataset_split / "Images"
        labels_dir = source_root / dataset_split / "Labels"
        if not images_dir.exists():
            print(f"[skip] {images_dir} not found")
            continue

        image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
        print(f"{dataset_split}: found {len(image_paths)} images")

        for img_path in image_paths:
            stem = img_path.stem
            label_path = labels_dir / f"{stem}.txt"
            if not label_path.exists():
                n_skipped += 1
                continue

            lines_out = []
            for line in label_path.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split()
                cls_id = int(parts[0])
                if cls_id not in VALID_CLASSES:
                    continue  # defensive -- should never happen on this dataset
                box = " ".join(parts[1:5])
                lines_out.append(f"{cls_id} {box}")

            if not lines_out:
                n_skipped += 1
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                n_skipped += 1
                continue

            dst_img = det_root / "images" / pipeline_split / img_path.name
            shutil.copy2(img_path, dst_img)
            (det_root / "labels" / pipeline_split / f"{stem}.txt").write_text("\n".join(lines_out))
            n_written += 1

    data_yaml = det_root.parent / "detection_mgi_data.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in CLASS_NAMES_7.items())
    data_yaml.write_text(
        f"path: {det_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n{names_block}\n"
    )

    print(f"\nWrote {n_written} images/labels (7-class MGI scheme), skipped {n_skipped}.")
    print(f"Detection root: {det_root}")
    print(f"Wrote {data_yaml}")
    print("\nNext: python3 scripts/train_detection_mgi.py "
          f"--data-yaml {data_yaml} --epochs 100")


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                         help="Path to the extracted Mendeley 'Dataset' folder (same one you used before)")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--det-output-dir", default=None,
                         help="Override output folder (default: sgsq_data/detection_mgi, derived from config)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.det_output_dir:
        det_root = Path(args.det_output_dir)
    else:
        base = Path(cfg["paths"]["detection_dataset"])  # e.g. sgsq_data/detection
        det_root = base.parent / f"{base.name}_mgi"      # -> sgsq_data/detection_mgi

    convert(args.source, det_root)
