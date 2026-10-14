"""
Single source of truth for IAR-to-severity thresholds (used by
data/synthetic_generator.py to keep synthetic label generation consistent).

Also provides a RULE-BASED, ZERO-PARAMETER severity classifier: apply these
exact thresholds directly to a segmentation model's predicted IAR, with no
learned severity head at all. This is the key diagnostic for explaining
ablation gaps: if this zero-parameter rule scores well, the learned
SeverityHead is underperforming a trivial baseline and the problem is in
Stage 3's learned mapping. If this rule ALSO scores badly, the problem is
upstream — segmentation noise in the predicted IAR itself.
"""

SEVERITY_CLASSES = [
    "healthy",
    "gingivitis",
    "mild_periodontitis",
    "moderate_periodontitis",
    "severe_periodontitis",
]

# IAR thresholds used both to generate synthetic ground-truth labels AND as
# the rule-based diagnostic classifier's decision boundaries. Arbitrary
# placeholders for synthetic data — replace with clinically validated
# thresholds once real data is available.
IAR_THRESHOLDS = [0.03, 0.12, 0.25, 0.45]


def iar_to_severity(iar: float) -> int:
    for idx, t in enumerate(IAR_THRESHOLDS):
        if iar < t:
            return idx
    return len(IAR_THRESHOLDS)


def rule_based_predict(iar_values: list) -> list:
    """Zero-parameter diagnostic classifier: threshold IAR directly, no learning."""
    return [iar_to_severity(v) for v in iar_values]


if __name__ == "__main__":
    test_values = [0.01, 0.05, 0.15, 0.30, 0.60]
    expected = [0, 1, 2, 3, 4]
    result = rule_based_predict(test_values)
    assert result == expected, f"expected {expected}, got {result}"
    print("Rule-based thresholding: OK")
