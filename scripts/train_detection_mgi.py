"""
scripts/train_detection_mgi.py
================================
Trains YOLOv8 on the 7-class (2 jaw-region + 5 per-tooth-MGI-level) detection
scheme written by data/convert_mendeley_dataset_mgi_detection.py. This is a
SEPARATE model from your existing tooth/gingiva detector -- it exists to make
per-tooth severity something the detector actually predicts, closing the gap
where box-derived severity features (ITF/IAF) previously required
ground-truth per-tooth MGI labels that a deployed system wouldn't have.

Distinguishing 5 fine-grained per-tooth appearance classes from a photo is a
strictly harder task than binary tooth/gingiva localization -- expect lower
mAP than your existing 2-class detector, especially on the middle MGI levels
(1/2/3), which are visually closer to each other than to the extremes
(healthy vs. severe). Report this mAP-per-class breakdown honestly; it is
itself informative (e.g. it tells you whether "moderate vs. high" MGI is
even visually separable at this image resolution).

Usage:
    python3 scripts/train_detection_mgi.py --data-yaml sgsq_data/detection_mgi_data.yaml
    python3 scripts/train_detection_mgi.py --data-yaml sgsq_data/detection_mgi_data.yaml --epochs 2   # smoke test
"""
import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


def main(data_yaml: str, config_path: str, epochs_override: int, base_model: str):
    data_yaml_path = Path(data_yaml)
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"{data_yaml_path} not found -- run data/convert_mendeley_dataset_mgi_detection.py first."
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    det_cfg = cfg["detection"]
    epochs = epochs_override or det_cfg["epochs"]

    model = YOLO(base_model)
    model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=det_cfg["img_size"],
        batch=det_cfg["batch_size"],
        project=cfg["paths"]["checkpoints"],
        name="detection_mgi",
        pretrained=True,
        patience=max(5, epochs),
        verbose=True,
    )

    metrics = model.val()
    print("Validation metrics (7-class MGI detector):", metrics.results_dict)
    print("\nPer-class results (check mAP for each tooth_mgi* class individually -- "
          "this tells you which severity levels the detector can actually separate):")
    if hasattr(metrics, "maps"):
        names = model.names
        for i, ap in enumerate(metrics.maps):
            print(f"  class {i} ({names.get(i, '?')}): mAP50-95 = {ap:.4f}")

    # IMPORTANT: ultralytics does NOT save under <project>/<name>/weights/best.pt
    # as a naive reading of the project=/name= args would suggest -- it nests
    # under a "runs/detect/" prefix (confirmed empirically; the real save_dir
    # is only known from the trainer after training finishes). Every other
    # checkpoint in this pipeline lives as a flat file directly under
    # checkpoints/, so we copy the real best.pt there for consistency and to
    # avoid every downstream script having to guess ultralytics' nesting.
    real_best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"
    flat_ckpt_dir = Path(cfg["paths"]["checkpoints"])
    flat_ckpt_dir.mkdir(parents=True, exist_ok=True)
    flat_best_pt = flat_ckpt_dir / "detection_mgi_best.pt"
    if real_best_pt.exists():
        shutil.copy2(real_best_pt, flat_best_pt)
        print(f"\nBest checkpoint (ultralytics' own save location): {real_best_pt}")
        print(f"Also copied to: {flat_best_pt}  (this is the path other scripts in this "
              f"patch default to -- use --detector-ckpt to override if you move it)")
    else:
        print(f"\n[warn] expected best.pt at {real_best_pt} but it wasn't found -- "
              f"check the 'Results saved to ...' line above for the actual location "
              f"and pass it explicitly via --detector-ckpt to the next script.")

    print("Next: python3 scripts/compute_detector_derived_features.py --config configs/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--base-model", default="yolov8n.pt")
    args = parser.parse_args()
    main(args.data_yaml, args.config, args.epochs, args.base_model)
