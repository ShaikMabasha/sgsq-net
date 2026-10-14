"""
Grad-CAM explainability, applied to:
  1. The EfficientNet-B0 baseline classifier
  2. SGSQ-Net's segmentation encoder

NOTE on ViT: standard Grad-CAM doesn't apply cleanly to a patch-token
transformer. Attention-rollout is the standard alternative but is a
separate implementation — documented as a known gap.

Usage:
    python scripts/explainability.py --config configs/config.yaml \\
        --model efficientnet --image path/to/img.jpg --out gradcam_cls.jpg

    python scripts/explainability.py --config configs/config.yaml \\
        --model segmentation --image path/to/img.jpg --out gradcam_seg.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.unet import ResNetUNet
from models.baselines.classifiers import EfficientNetB0Classifier
from data.baseline_dataset import get_classification_transforms
from data.dataset import get_segmentation_transforms


class GradCAM:
    """Generic Grad-CAM via forward/backward hooks on a chosen target layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, input_tensor: torch.Tensor, target_scalar_fn):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = target_scalar_fn(output)
        target.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam[0, 0].cpu().numpy()

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        target_h, target_w = input_tensor.shape[-2:]
        cam_resized = cv2.resize(cam, (target_w, target_h))
        return cam_resized


def overlay_heatmap(image_rgb_uint8: np.ndarray, heatmap01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap_color = cv2.applyColorMap((heatmap01 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (image_rgb_uint8.astype(np.float32) * (1 - alpha) + heatmap_color.astype(np.float32) * alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def run_classifier_gradcam(cfg: dict, image_path: str, out_path: str):
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    class_names = cfg["severity"]["classes"]
    ckpt_path = Path(cfg["paths"]["checkpoints"]) / "baselines" / "efficientnet_classifier_best.pt"

    model = EfficientNetB0Classifier(num_classes=len(class_names)).to(device)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"[warn] no checkpoint at {ckpt_path}; using untrained weights (Grad-CAM will be meaningless)")
    model.eval()

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transform = get_classification_transforms(224, train=False)
    tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)

    gradcam = GradCAM(model, model.get_target_layer())
    heatmap = gradcam.compute(tensor, lambda out: out[0, out.argmax(dim=1).item()])

    resized_rgb = cv2.resize(image_rgb, (224, 224))
    overlay = overlay_heatmap(resized_rgb, heatmap)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved classifier Grad-CAM to {out_path}")


def run_segmentation_gradcam(cfg: dict, image_path: str, out_path: str):
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    seg_cfg = cfg["segmentation"]
    ckpt_path = Path(cfg["paths"]["checkpoints"]) / "segmentation_best.pt"

    model = ResNetUNet(num_classes=seg_cfg["num_classes"]).to(device)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"[warn] no checkpoint at {ckpt_path}; using untrained weights (Grad-CAM will be meaningless)")
    model.eval()

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_size = seg_cfg["img_size"]
    transform = get_segmentation_transforms(img_size, train=False)
    tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)

    gradcam = GradCAM(model, model.enc4)
    heatmap = gradcam.compute(tensor, lambda out: out.sum())

    resized_rgb = cv2.resize(image_rgb, (img_size, img_size))
    overlay = overlay_heatmap(resized_rgb, heatmap)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved segmentation Grad-CAM to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", choices=["efficientnet", "segmentation"], required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="gradcam_output.jpg")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model == "efficientnet":
        run_classifier_gradcam(cfg, args.image, args.out)
    else:
        run_segmentation_gradcam(cfg, args.image, args.out)
