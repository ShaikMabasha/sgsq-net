"""
Full SGSQ-Net inference on a single intraoral photo:
    1. DETECT   -> YOLOv8 locates tooth/gingiva ROIs
    2. SEGMENT  -> U-Net produces an inflamed-gingiva mask per ROI
    3. QUANTIFY -> IAR + mean-redness computed from the mask, fed to the
                    ordinal Severity Head -> final severity class

Usage:
    python scripts/inference_pipeline.py --image path/to/photo.jpg --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.unet import ResNetUNet
from models.severity_head import SeverityHead, FeatureNormalizer, compute_iar, compute_mean_redness
from data.dataset import get_segmentation_transforms


class SGSQPipeline:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device(self.cfg["device"] if torch.cuda.is_available() else "cpu")
        ckpt_dir = Path(self.cfg["paths"]["checkpoints"])

        det_weights = ckpt_dir / "detection" / "weights" / "best.pt"
        self.detector = YOLO(str(det_weights) if det_weights.exists() else self.cfg["detection"]["model"])

        seg_cfg = self.cfg["segmentation"]
        self.segmenter = ResNetUNet(num_classes=seg_cfg["num_classes"]).to(self.device)
        seg_ckpt = ckpt_dir / "segmentation_best.pt"
        if seg_ckpt.exists():
            self.segmenter.load_state_dict(torch.load(seg_ckpt, map_location=self.device))
        else:
            print(f"[warn] no trained segmentation checkpoint at {seg_ckpt}; using untrained weights")
        self.segmenter.eval()
        self.seg_img_size = seg_cfg["img_size"]
        self.seg_transform = get_segmentation_transforms(seg_cfg["img_size"], train=False)

        sev_cfg = self.cfg["severity"]
        self.class_names = sev_cfg["classes"]
        self.severity_head = SeverityHead(num_classes=len(self.class_names), input_dim=2).to(self.device)
        sev_ckpt = ckpt_dir / "severity_head_best.pt"
        if sev_ckpt.exists():
            self.severity_head.load_state_dict(torch.load(sev_ckpt, map_location=self.device))
        else:
            print(f"[warn] no trained severity-head checkpoint at {sev_ckpt}; using untrained weights")
        self.severity_head.eval()

        # Apply the SAME feature normalization used at training time
        self.feature_normalizer = FeatureNormalizer()
        norm_stats_path = ckpt_dir / "severity_feature_stats.json"
        if norm_stats_path.exists():
            self.feature_normalizer.load(norm_stats_path)
        else:
            print(f"[warn] no feature normalizer stats at {norm_stats_path}; "
                  f"features will be used UNNORMALIZED")

    def _segment_and_score(self, crop_rgb_float01: np.ndarray):
        orig_h, orig_w = crop_rgb_float01.shape[:2]
        crop_uint8 = (crop_rgb_float01 * 255).astype(np.uint8)

        tensor = self.seg_transform(image=crop_uint8)["image"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.segmenter(tensor)
            probs = torch.sigmoid(logits)

            iar = compute_iar(probs).item()
            img_resized = torch.nn.functional.interpolate(
                torch.from_numpy(crop_rgb_float01).permute(2, 0, 1).unsqueeze(0).to(self.device),
                size=probs.shape[-2:], mode="bilinear", align_corners=False,
            )
            redness = compute_mean_redness(img_resized, probs).item()

            features_raw = torch.tensor([[iar, redness]], dtype=torch.float32, device=self.device)
            if self.feature_normalizer.mean is not None:
                features = self.feature_normalizer.transform(features_raw)
            else:
                features = features_raw
            sev_logits = self.severity_head(features)
            sev_pred = SeverityHead.logits_to_class(sev_logits).item()

        mask_np = (probs > 0.5).float().cpu().numpy()[0, 0]
        mask_resized = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return (mask_resized * 255).astype(np.uint8), sev_pred, {"iar": iar, "mean_redness": redness}

    def run(self, image_path: str, conf_threshold: float = 0.25) -> list:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image_rgb_float = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        results = self.detector.predict(image_path, conf=conf_threshold, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det_class_idx = int(box.cls[0].item())
            det_conf = float(box.conf[0].item())
            det_label = results.names[det_class_idx]

            if det_label != "gingiva":
                detections.append({
                    "bbox": [x1, y1, x2, y2], "detected_region": det_label,
                    "detection_confidence": det_conf, "severity": None,
                })
                continue

            crop = image_rgb_float[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            mask, sev_pred, features = self._segment_and_score(crop)
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "detected_region": det_label,
                "detection_confidence": det_conf,
                "severity": self.class_names[sev_pred],
                "severity_index": sev_pred,  # ordinal position, works regardless of class-name taxonomy
                "iar": features["iar"],
                "mean_redness": features["mean_redness"],
                "mask": mask,
            })

        return detections

    def visualize(self, image_path: str, detections: list, save_path: str = "output_visualization.jpg"):
        image = cv2.imread(image_path)
        overlay = image.copy()

        # BUG FIX: colors keyed by ordinal severity INDEX, not hardcoded
        # class-name strings -- works regardless of taxonomy naming.
        num_classes = len(self.class_names) if hasattr(self, "class_names") else 5
        ordinal_colors = [
            tuple(int(c) for c in ((0, 200, 0), (0, 200, 200), (0, 140, 255), (0, 80, 255), (0, 0, 255))[
                min(i, 4)
            ]) for i in range(max(num_classes, 1))
        ]

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            sev_idx = det.get("severity_index")
            color = ordinal_colors[sev_idx] if sev_idx is not None and sev_idx < len(ordinal_colors) else (200, 200, 200)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            if det.get("severity") is not None:
                label = f"{det['severity']} (IAR={det['iar']:.2f})"
                cv2.putText(image, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                mask = det["mask"]
                region = overlay[y1:y2, x1:x2]
                if region.shape[:2] == mask.shape:
                    red_layer = np.zeros_like(region)
                    red_layer[:, :, 2] = mask
                    overlay[y1:y2, x1:x2] = cv2.addWeighted(region, 0.7, red_layer, 0.3, 0)

        blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        cv2.imwrite(save_path, blended)
        print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out", default="output_visualization.jpg")
    args = parser.parse_args()

    pipeline = SGSQPipeline(args.config)
    detections = pipeline.run(args.image)

    for i, det in enumerate(detections):
        print(f"\nRegion {i+1}: {det['detected_region']} (det_conf={det['detection_confidence']:.2f})")
        if det.get("severity") is not None:
            print(f"  Severity: {det['severity']} | IAR={det['iar']:.3f} | mean_redness={det['mean_redness']:.3f}")

    pipeline.visualize(args.image, detections, save_path=args.out)
