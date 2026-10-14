"""
scripts/check_detector_perclass.py
==================================
Re-validates the final MGI detector on the TEST split and prints the per-class table that appears in the
paper (Precision, Recall, mAP@50, mAP@50:95), so the numbers can be checked row by row.

Usage (from the sgsq_pipeline root; find your dataset yaml first):
    !find . -name "*.yaml" -not -path "*/checkpoints/*" -not -path "*/.ipynb_checkpoints/*" | head -20
    python3 scripts/check_detector_perclass.py --data <path/to/detection_data.yaml> --config configs/config.yaml
Nothing is trained; GPU optional. Uses the Ultralytics defaults (the same as the paper's P/R/mAP).
"""
import argparse
from pathlib import Path


def main():
    import yaml
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ultralytics dataset yaml of the 7-class detector")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--detector-ckpt", default="detection_mgi_best.pt")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    ckpt = Path(cfg["paths"]["checkpoints"]) / a.detector_ckpt
    m = YOLO(str(ckpt)).val(data=a.data, split="test", plots=False, verbose=False)
    b = m.box
    print(f"\n{'class':28s} {'P':>7s} {'R':>7s} {'mAP50':>7s} {'mAP50-95':>9s}")
    for i, c in enumerate(b.ap_class_index):
        print(f"{m.names[int(c)]:28s} {b.p[i]:7.3f} {b.r[i]:7.3f} {b.ap50[i]:7.3f} {b.ap[i]:9.3f}")
    print(f"{'ALL (mean over classes)':28s} {b.mp:7.3f} {b.mr:7.3f} {b.map50:7.3f} {b.map:9.3f}")


if __name__ == "__main__":
    main()
