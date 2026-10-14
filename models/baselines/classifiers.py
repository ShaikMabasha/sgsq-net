"""
Baseline END-TO-END severity classifiers — predict severity directly from
raw pixels, with no segmentation stage in between. These serve two purposes:

  1. BENCHMARK comparison against literature architectures (EfficientNet-B0
     represents the common single-CNN-backbone approach; ViT-B/16
     represents the attention-based approach, e.g. AB-MIML-style methods).

  2. ABLATION baseline — comparing these against SGSQ-Net's
     segmentation-guided severity head tells you whether the
     segmentation-derived-feature approach actually adds value over a
     conventional black-box classifier.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class EfficientNetB0Classifier(nn.Module):
    """Representative of single-CNN-backbone literature baselines."""

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.net = tv_models.efficientnet_b0(weights=weights)
        in_features = self.net.classifier[-1].in_features
        self.net.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)

    def get_target_layer(self):
        """Last conv layer, for Grad-CAM."""
        return self.net.features[-1]


class ViTClassifier(nn.Module):
    """Representative of attention/transformer-based literature baselines."""

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.net = tv_models.vit_b_16(weights=weights)
        in_features = self.net.heads.head.in_features
        self.net.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)

    # NOTE: Grad-CAM in the usual conv-activation sense doesn't apply to ViT's
    # patch-token representation. Attention-rollout is the standard alternative
    # but is a separate, more involved implementation than Grad-CAM -- left as
    # documented future work. Use EfficientNetB0Classifier's Grad-CAM instead.


if __name__ == "__main__":
    torch.manual_seed(0)

    model = EfficientNetB0Classifier(num_classes=5)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("EfficientNetB0Classifier output shape:", out.shape)
    assert out.shape == (2, 5)
    print("OK")

    vit = ViTClassifier(num_classes=5)
    dummy_vit = torch.randn(2, 3, 224, 224)
    out_vit = vit(dummy_vit)
    print("ViTClassifier output shape:", out_vit.shape)
    assert out_vit.shape == (2, 5)
    print("OK")
