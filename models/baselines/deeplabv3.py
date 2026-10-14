"""
DeepLabV3+ baseline segmentation model — one of the most common architectures
in the periodontal-photo segmentation literature, used here as a benchmark
comparison against SGSQ-Net's ResNet-U-Net encoder.

Wraps torchvision's DeepLabV3 (ResNet50 backbone) to match ResNetUNet's
interface: same input shape in, same (B, num_classes, H, W) logits out, so
it's a drop-in replacement for benchmarking with the same training loop,
loss function, and evaluation metrics as models/unet.py.
"""

import torch
import torch.nn as nn
import torchvision.models.segmentation as seg_models


class DeepLabV3Baseline(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        weights = seg_models.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        self.model = seg_models.deeplabv3_resnet50(weights=weights)
        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        if self.model.aux_classifier is not None:
            aux_in = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Conv2d(aux_in, num_classes, kernel_size=1)

    def forward(self, x):
        out = self.model(x)
        return out["out"]


if __name__ == "__main__":
    model = DeepLabV3Baseline(num_classes=1)
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    print("Output shape:", out.shape)
    assert out.shape == (2, 1, 256, 256), "Unexpected output shape!"
    print("OK")
