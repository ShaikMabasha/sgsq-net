"""
Dataset classes for the segmentation and severity-quantification stages.

Detection uses YOLO's own format directly via ultralytics — no custom
Dataset class needed there.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_segmentation_transforms(img_size: int, train: bool = True):
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class SegmentationDataset(Dataset):
    """Image + binary mask pairs. Expects images/*.jpg and masks/*.png with matching stems."""

    def __init__(self, root_dir: str, filenames: list = None, img_size: int = 256, train: bool = True):
        self.images_dir = Path(root_dir) / "images"
        self.masks_dir = Path(root_dir) / "masks"
        self.transform = get_segmentation_transforms(img_size, train)

        if filenames is not None:
            self.image_paths = [self.images_dir / f for f in filenames]
        else:
            self.image_paths = sorted(
                [p for ext in ("*.jpg", "*.jpeg", "*.png") for p in self.images_dir.glob(ext)]
            )
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found under {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _find_mask_path(self, image_path: Path) -> Path:
        stem = image_path.stem
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = self.masks_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No mask found for image {image_path.name} in {self.masks_dir}")

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self._find_mask_path(img_path)

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        augmented = self.transform(image=image, mask=mask)
        img_t = augmented["image"]
        mask_t = augmented["mask"].unsqueeze(0).float()
        return img_t, mask_t, img_path.name


class SeverityFeatureDataset(Dataset):
    """
    Reads a manifest CSV (filename, split, severity_label, ...) and computes
    [IAR, mean_redness] features either from:

      - mask_source="ground_truth": the synthetic/annotated masks directly
        (fast, but doesn't match what inference sees).
      - mask_source="predicted": a trained segmentation model's own output
        (slower to build, but this is what fixes the train/inference
        mismatch bug — Stage 3 must learn on the same kind of noisy masks
        it will receive at real inference time, not the clean ground truth
        it will never see again after training).
    """

    def __init__(self, csv_path: str, seg_root: str, split: str = "train",
                 mask_source: str = "ground_truth", segmentation_model=None,
                 device=None, seg_img_size: int = 256):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.images_dir = Path(seg_root) / "images"
        self.masks_dir = Path(seg_root) / "masks"
        self.mask_source = mask_source

        if mask_source == "predicted":
            if segmentation_model is None or device is None:
                raise ValueError("mask_source='predicted' requires segmentation_model and device")
            self._features = self._compute_predicted_features(segmentation_model, device, seg_img_size)
        elif mask_source == "ground_truth":
            self._features = self._compute_ground_truth_features()
        else:
            raise ValueError(f"Unknown mask_source '{mask_source}'")

    def _compute_ground_truth_features(self):
        features = []
        for _, row in self.df.iterrows():
            stem = Path(row["filename"]).stem
            image = cv2.imread(str(self.images_dir / f"{stem}.jpg"))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mask = cv2.imread(str(self.masks_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
            features.append(self._iar_and_redness(image, mask))
        return features

    def _compute_predicted_features(self, model, device, seg_img_size):
        transform = get_segmentation_transforms(seg_img_size, train=False)
        features = []
        model.eval()
        with torch.no_grad():
            for _, row in self.df.iterrows():
                stem = Path(row["filename"]).stem
                image_bgr = cv2.imread(str(self.images_dir / f"{stem}.jpg"))
                image_rgb_uint8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_float = image_rgb_uint8.astype(np.float32) / 255.0

                tensor = transform(image=image_rgb_uint8)["image"].unsqueeze(0).to(device)
                logits = model(tensor)
                probs = torch.sigmoid(logits)  # (1, 1, h, w) at seg_img_size
                pred_mask = (probs > 0.5).float().cpu().numpy()[0, 0]

                pred_mask_full = cv2.resize(
                    pred_mask, (image_float.shape[1], image_float.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                features.append(self._iar_and_redness(image_float, pred_mask_full))
        return features

    @staticmethod
    def _iar_and_redness(image_float01: np.ndarray, mask01: np.ndarray):
        iar = float(mask01.mean())
        redness_map = image_float01[:, :, 0] - 0.5 * (image_float01[:, :, 1] + image_float01[:, :, 2])
        inflamed_px = mask01 > 0
        mean_redness = float(redness_map[inflamed_px].mean()) if inflamed_px.sum() > 0 else 0.0
        return iar, mean_redness

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        iar, mean_redness = self._features[idx]
        row = self.df.iloc[idx]
        features = torch.tensor([iar, mean_redness], dtype=torch.float32)
        label = torch.tensor(int(row["severity_label"]), dtype=torch.long)
        return features, label
