"""
Severity quantification module (Stage 3 — the core novelty of SGSQ-Net).

Rather than classifying severity directly from raw pixels, this module takes
quantitative features COMPUTED FROM THE SEGMENTATION MASK (Inflamed Area
Ratio, and optionally a mean-redness color feature) and maps them to an
ordinal severity class. This keeps the severity decision traceable to a
measurable quantity instead of a black-box embedding.

Ordinal regression is used (via the CORAL / cumulative-logit style approach)
rather than plain softmax classification, because severity classes are
inherently ordered (healthy < gingivitis < mild < moderate < severe) — a
model that confuses "healthy" and "severe" should be penalized more than one
that confuses adjacent classes, which plain cross-entropy does not capture.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn


class FeatureNormalizer:
    """
    Fixes bug: IAR is bounded [0,1] but mean_redness has a very different,
    unnormalized range. Feeding both raw into the severity head starves the
    optimizer. Fit on the TRAIN split only, then apply the same fixed stats
    everywhere (train/val/test/inference) to avoid leakage.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, features: torch.Tensor):
        """features: (N, D) tensor."""
        self.mean = features.mean(dim=0)
        self.std = features.std(dim=0).clamp(min=1e-6)  # avoid divide-by-zero on constant features
        return self

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            raise RuntimeError("FeatureNormalizer.fit() must be called (or load()) before transform().")
        device = features.device
        return (features - self.mean.to(device)) / self.std.to(device)

    def fit_transform(self, features: torch.Tensor) -> torch.Tensor:
        return self.fit(features).transform(features)

    def save(self, path: str):
        Path(path).write_text(json.dumps({
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }))

    def load(self, path: str):
        d = json.loads(Path(path).read_text())
        self.mean = torch.tensor(d["mean"], dtype=torch.float32)
        self.std = torch.tensor(d["std"], dtype=torch.float32)
        return self


def compute_iar(mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Inflamed Area Ratio: fraction of pixels in the mask classified as inflamed.

    Args:
        mask: (B, 1, H, W) tensor of sigmoid probabilities OR binary mask.
    Returns:
        (B,) tensor of IAR values in [0, 1].
    """
    binary = (mask > threshold).float()
    return binary.mean(dim=(1, 2, 3))


def compute_mean_redness(image: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Mean redness (R channel minus G/B average) within the inflamed region only.
    Used as a secondary feature alongside IAR — inflammation tends to be both
    larger in area AND more intensely red as severity increases.

    Args:
        image: (B, 3, H, W) RGB tensor, values in [0, 1]
        mask:  (B, 1, H, W) sigmoid probabilities or binary mask
    Returns:
        (B,) tensor; 0 if no inflamed pixels present in a sample.
    """
    binary = (mask > threshold).float()
    redness_map = image[:, 0:1] - 0.5 * (image[:, 1:2] + image[:, 2:3])  # (B, 1, H, W)
    masked_sum = (redness_map * binary).sum(dim=(1, 2, 3))
    pixel_count = binary.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return masked_sum / pixel_count


class SeverityHead(nn.Module):
    """
    Small MLP mapping [IAR, mean_redness] (or just IAR) -> ordinal severity logits.

    Uses the CORAL framework: K-1 binary classifiers sharing weights, each
    predicting P(severity > k) for k = 0..K-2. This guarantees monotonic,
    consistent ordinal predictions (unlike independent per-threshold heads).
    """

    def __init__(self, num_classes: int = 5, input_dim: int = 2, hidden_dim: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Single shared weight vector -> scalar, then K-1 learned biases (CORAL trick)
        self.shared_weight = nn.Linear(hidden_dim, 1, bias=False)
        self.thresholds = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, input_dim) — e.g. torch.stack([iar, mean_redness], dim=1)
        Returns:
            (B, num_classes - 1) logits, one per P(severity > k) cumulative threshold
        """
        h = self.backbone(features)
        base_logit = self.shared_weight(h)  # (B, 1)
        return base_logit + self.thresholds.unsqueeze(0)  # (B, K-1)

    @staticmethod
    def logits_to_class(logits: torch.Tensor) -> torch.Tensor:
        """Convert cumulative-threshold logits to a single predicted class index."""
        probs = torch.sigmoid(logits)  # P(severity > k) for each k
        # Predicted class = number of thresholds exceeded
        preds = (probs > 0.5).sum(dim=1)
        return preds


class CoralLoss(nn.Module):
    """CORAL ordinal loss: binary cross-entropy against K-1 cumulative labels."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Build cumulative targets: target_k = 1 if label > k else 0
        levels = torch.arange(self.num_classes - 1, device=labels.device).unsqueeze(0)
        cumulative_targets = (labels.unsqueeze(1) > levels).float()  # (B, K-1)
        return self.bce(logits, cumulative_targets)


if __name__ == "__main__":
    torch.manual_seed(0)
    head = SeverityHead(num_classes=5, input_dim=2)
    loss_fn = CoralLoss(num_classes=5)

    dummy_features = torch.rand(8, 2)
    dummy_labels = torch.randint(0, 5, (8,))

    logits = head(dummy_features)
    print("Logits shape:", logits.shape)
    assert logits.shape == (8, 4)

    loss = loss_fn(logits, dummy_labels)
    print("Loss:", loss.item())

    preds = SeverityHead.logits_to_class(logits)
    print("Predicted classes:", preds)
    assert preds.shape == (8,)
    print("OK")
