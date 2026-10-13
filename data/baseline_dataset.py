"""
Dataset for training the baseline end-to-end classifiers (EfficientNet-B0,
ViT-B/16) directly on raw images against severity labels — no segmentation
involved. Reads the SAME severity_labels.csv used by SGSQ-Net's Stage 3, so
splits are identical across all models for a fair benchmark/ablation
comparison.
"""

from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_classification_transforms(img_size: int, train: bool = True):
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


class RawImageSeverityDataset(Dataset):
    """Raw image -> severity label, for end-to-end baseline classifiers."""

    def __init__(self, csv_path: str, images_root: str, split: str = "train",
                 img_size: int = 224, train: bool = True):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.images_dir = Path(images_root)
        self.transform = get_classification_transforms(img_size, train)

        if len(self.df) == 0:
            raise RuntimeError(f"No rows found for split='{split}' in {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        stem = Path(row["filename"]).stem
        image = cv2.imread(str(self.images_dir / f"{stem}.jpg"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=image)
        label = torch.tensor(int(row["severity_label"]), dtype=torch.long)
        return augmented["image"], label
