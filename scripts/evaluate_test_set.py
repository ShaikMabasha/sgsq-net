"""
scripts/evaluate_test_set.py
=============================
Held-out TEST-SET evaluation with statistical rigor, for eliminating the
single biggest "reviewer can reject this" loophole in the project: every
number reported so far (Dice, QWK, accuracy) came from the VALIDATION
split, which was ALSO used for early-stopping checkpoint selection. This
script evaluates every trained model exactly once on the 182 Mendeley
`split=="test"` images/rows that no training or model-selection step has
ever touched, and reports:

  1. A point estimate on TEST.
  2. A bootstrap 95% confidence interval around it.
  3. For severity models, a paired bootstrap significance test of the
     PRIMARY SGSQ-Net severity approach against every baseline, on the
     SAME test items.

DESIGN NOTE -- this script deliberately reuses the project's own dataset
and model classes verbatim (SegmentationDataset, SeverityFeatureDataset,
RawImageSeverityDataset, ResNetUNet, DeepLabV3Baseline, SeverityHead,
FeatureNormalizer, RealBoxFeatureDataset, per_sample_dice) instead of
re-implementing image loading/preprocessing/thresholding from scratch.
Re-implementing those independently risks *introducing* a new
train/eval preprocessing mismatch -- exactly the class of bug that
produced the earlier benchmark_compare.py discrepancy (see
README_TEST_EVAL.md for that diagnosis). Reusing the exact same code
path, pointed at the test split instead of val, is the safer fix.

SECOND FIX included here: `models/rule_based_severity.py`'s
`rule_based_predict()` applies IAR thresholds (0.03, 0.12, 0.25, 0.45)
that were calibrated for the SYNTHETIC data generator, not for Mendeley's
real feature distribution. Applying stale synthetic-era thresholds to a
real dataset's features is itself a reviewer-flaggable loophole (it can
produce an arbitrarily good- or bad-looking "diagnostic" baseline with no
real meaning). This script recalibrates the rule-based baseline's
thresholds on the TRAIN split's own quantiles (same style already used,
correctly, in scripts/train_severity_real_interim.py) before applying it
to TEST, for both the predicted-mask IAR and the box-derived ITF/IAF
features.

Usage (from the sgsq_pipeline root):

    python3 scripts/evaluate_test_set.py --config configs/config.yaml

Run with --help to see every path/checkpoint-name override.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stats_utils import bootstrap_ci, compute_qwk_bootstrap, paired_qwk_bootstrap_test  # noqa: E402

from data.dataset import SegmentationDataset, SeverityFeatureDataset  # noqa: E402
from data.baseline_dataset import RawImageSeverityDataset  # noqa: E402
from models.unet import ResNetUNet  # noqa: E402
from models.baselines.deeplabv3 import DeepLabV3Baseline  # noqa: E402
from models.baselines.classifiers import EfficientNetB0Classifier, ViTClassifier  # noqa: E402
from models.severity_head import SeverityHead, FeatureNormalizer  # noqa: E402
from scripts.train_segmentation import per_sample_dice  # noqa: E402
from scripts.train_severity_real_interim import RealBoxFeatureDataset  # noqa: E402


# ==========================================================================
# Segmentation
# ==========================================================================

def per_sample_iou(logits, targets, threshold: float = 0.5, smooth: float = 1.0):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    has_positive = targets_flat.sum(dim=1) > 0
    return iou, has_positive


@torch.no_grad()
def evaluate_segmentation_on_test(model, loader, device) -> dict:
    model.eval()
    all_dice, all_iou, all_pos = [], [], []
    for images, masks, _names in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        dice, has_pos = per_sample_dice(logits, masks)
        iou, _ = per_sample_iou(logits, masks)
        all_dice.append(dice.cpu().numpy())
        all_iou.append(iou.cpu().numpy())
        all_pos.append(has_pos.cpu().numpy())
    all_dice = np.concatenate(all_dice)
    all_iou = np.concatenate(all_iou)
    all_pos = np.concatenate(all_pos)
    return {
        "dice_overall": all_dice,
        "iou_overall": all_iou,
        "dice_positive_only": all_dice[all_pos],
        "iou_positive_only": all_iou[all_pos],
        "frac_positive": float(np.mean(all_pos)),
        "n": len(all_dice),
    }


def run_segmentation_eval(cfg, ckpt_dir: Path, device, n_boot: int, sgsq_ckpt_name: str, deeplab_ckpt_name: str) -> dict:
    print("\n" + "=" * 70)
    print("SEGMENTATION -- HELD-OUT TEST SPLIT")
    print("=" * 70)

    seg_cfg = cfg["segmentation"]
    seg_root = cfg["paths"]["segmentation_dataset"]
    test_files = sorted([p.name for p in (Path(seg_root) / "images").glob("*.jpg") if p.name.startswith("test_")])
    if len(test_files) == 0:
        print(f"[skip] no test_*.jpg images found under {seg_root}/images -- has data/convert_mendeley_dataset.py "
              f"been run, and does it write a test split? Skipping segmentation TEST evaluation.")
        return {}

    seg_ds = SegmentationDataset(seg_root, filenames=test_files, img_size=seg_cfg["img_size"], train=False)
    seg_loader = DataLoader(seg_ds, batch_size=seg_cfg["batch_size"], shuffle=False)
    print(f"Test segmentation images found: {len(seg_ds)}")

    report = {}
    for model_name, model_cls, ckpt_path in [
        ("SGSQ-Net (ResNet-U-Net)", ResNetUNet, ckpt_dir / sgsq_ckpt_name),
        ("DeepLabV3+ (baseline)", DeepLabV3Baseline, ckpt_dir / "baselines" / deeplab_ckpt_name),
    ]:
        if not ckpt_path.exists():
            print(f"[skip] {model_name}: checkpoint not found at {ckpt_path}")
            continue
        model = model_cls(num_classes=seg_cfg["num_classes"]).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        result = evaluate_segmentation_on_test(model, seg_loader, device)
        dice_ci = bootstrap_ci(result["dice_positive_only"], np.mean, n_boot=n_boot)
        iou_ci = bootstrap_ci(result["iou_positive_only"], np.mean, n_boot=n_boot)
        dice_overall_ci = bootstrap_ci(result["dice_overall"], np.mean, n_boot=n_boot)

        print(f"\n{model_name}  (n={result['n']}, frac_positive={result['frac_positive']:.2f})")
        print(f"  Dice (positive-only) = {dice_ci['point']:.4f}  [95% CI {dice_ci['ci_low']:.4f}, {dice_ci['ci_high']:.4f}]")
        print(f"  IoU  (positive-only) = {iou_ci['point']:.4f}  [95% CI {iou_ci['ci_low']:.4f}, {iou_ci['ci_high']:.4f}]")
        print(f"  Dice (overall, all test images) = {dice_overall_ci['point']:.4f}  "
              f"[95% CI {dice_overall_ci['ci_low']:.4f}, {dice_overall_ci['ci_high']:.4f}]")
        if abs(result["frac_positive"] - 1.0) > 1e-6:
            print(f"  [diagnostic] frac_positive={result['frac_positive']:.2f} != 1.00 -- if every Mendeley image "
                  f"should contain at least one inflamed tooth (as during training), this indicates the test split "
                  f"folder may contain images from a DIFFERENT/contaminated source. Positive-only and overall Dice "
                  f"diverging further confirms this -- investigate sgsq_data/segmentation/images before trusting "
                  f"these numbers.")
        report[model_name] = {
            "dice_positive_only": dice_ci, "iou_positive_only": iou_ci, "dice_overall": dice_overall_ci,
            "frac_positive": result["frac_positive"], "n": result["n"],
        }
    return report


# ==========================================================================
# Severity
# ==========================================================================
def _qwk_fast(y, p, k):
    O = np.bincount(y * k + p, minlength=k * k).reshape(k, k).astype(float)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / (k - 1) ** 2
    return 1.0 - (w * O).sum() / (w * E).sum()


def paired_qwk_permutation_test(y_true, pred_a, pred_b, num_classes, n_perm=20000, seed=42):
    """Paired permutation test: under H0 the two models are exchangeable on each image,
    so their predictions are swapped independently with probability 0.5."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true, int); a = np.asarray(pred_a, int); b = np.asarray(pred_b, int)
    obs = _qwk_fast(y, a, num_classes) - _qwk_fast(y, b, num_classes)
    count = 0
    for _ in range(n_perm):
        s = rng.random(len(y)) < 0.5
        a2 = np.where(s, b, a); b2 = np.where(s, a, b)
        d = _qwk_fast(y, a2, num_classes) - _qwk_fast(y, b2, num_classes)
        count += abs(d) >= abs(obs) - 1e-12
    return {"diff": float(obs), "p_perm": float((count + 1) / (n_perm + 1))}


def holm_adjust(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def qwk_report(name, y_true, y_pred, num_classes, n_boot):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    qwk_ci = compute_qwk_bootstrap(np.asarray(y_true), np.asarray(y_pred), num_classes, n_boot=n_boot)
    print(f"\n{name}")
    print(f"  accuracy = {acc:.4f}, macro-F1 = {f1m:.4f}")
    print(f"  QWK = {qwk_ci['point']:.4f}  [95% CI {qwk_ci['ci_low']:.4f}, {qwk_ci['ci_high']:.4f}]  (n={qwk_ci['n']})")
    return {"accuracy": acc, "macro_f1": f1m, "qwk": qwk_ci}


def recalibrated_rule_based(train_vals: np.ndarray, test_vals: np.ndarray, num_classes: int) -> np.ndarray:
    """Quantile thresholds fit on TRAIN only (never val/test), applied to TEST.
    Mirrors the (correct) approach already used for the box-feature rule-based
    diagnostic in scripts/train_severity_real_interim.py -- unlike
    models/rule_based_severity.py's fixed synthetic-era thresholds, which are
    NOT valid for Mendeley's real feature distribution and should not be used
    for real-data comparisons.
    """
    quantiles = np.linspace(0, 1, num_classes + 1)[1:-1]
    edges = np.quantile(train_vals, quantiles)
    preds = np.digitize(test_vals, edges).clip(0, num_classes - 1)
    return preds


def run_severity_eval(cfg, ckpt_dir: Path, device, n_boot: int, args) -> dict:
    print("\n" + "=" * 70)
    print("SEVERITY -- HELD-OUT TEST SPLIT")
    print("=" * 70)

    sev_cfg = cfg["severity"]
    class_names = sev_cfg["classes"]
    num_classes = len(class_names)
    seg_cfg = cfg["segmentation"]
    seg_root = cfg["paths"]["segmentation_dataset"]
    csv_path = Path(cfg["paths"]["severity_dataset"]) / "severity_labels.csv"
    images_root = Path(seg_root) / "images"

    preds_by_model: dict[str, np.ndarray] = {}
    y_true_ref = None
    report = {}

    # ---- (a) SGSQ-Net segmentation-guided severity (the paper's primary
    #          claimed architecture: Detection -> Segmentation -> Severity) ----
    sgsq_seg_ckpt = ckpt_dir / args.sgsq_seg_ckpt
    sgsq_sev_ckpt = ckpt_dir / args.sgsq_sev_ckpt
    norm_stats_path = ckpt_dir / args.sgsq_sev_normalizer
    seg_guided_available = sgsq_seg_ckpt.exists() and sgsq_sev_ckpt.exists() and norm_stats_path.exists()

    seg_model = None
    if seg_guided_available:
        seg_model = ResNetUNet(num_classes=seg_cfg["num_classes"]).to(device)
        seg_model.load_state_dict(torch.load(sgsq_seg_ckpt, map_location=device))
        seg_model.eval()

        sev_model = SeverityHead(num_classes=num_classes, input_dim=2).to(device)
        sev_model.load_state_dict(torch.load(sgsq_sev_ckpt, map_location=device))
        sev_model.eval()

        normalizer = FeatureNormalizer().load(norm_stats_path)

        print("  [predicted-IAR] running segmentation model over the 182 TEST images "
              "(one image at a time, no progress bar in the underlying dataset class -- "
              "this is expected to take roughly a minute, not a hang)...")
        test_ds = SeverityFeatureDataset(csv_path, seg_root, split="test", mask_source="predicted",
                                          segmentation_model=seg_model, device=device, seg_img_size=seg_cfg["img_size"])
        print("  [predicted-IAR] TEST features done.")
        preds, labels = [], []
        with torch.no_grad():
            for i in range(len(test_ds)):
                features_raw, label = test_ds[i]
                features = normalizer.transform(features_raw.unsqueeze(0).to(device))
                logits = sev_model(features)
                preds.append(SeverityHead.logits_to_class(logits).item())
                labels.append(label.item())
        preds = np.array(preds)
        labels = np.array(labels)
        y_true_ref = labels
        name = "SGSQ-Net (segmentation-guided severity)"
        preds_by_model[name] = preds
        report[name] = qwk_report(name, labels, preds, num_classes, n_boot)

        # Recalibrated rule-based diagnostic on the SAME predicted-IAR feature,
        # thresholds fit on TRAIN split's predicted-IAR distribution (not the
        # stale synthetic-era fixed thresholds in models/rule_based_severity.py).
        # NOTE: this reruns the segmentation model over all ~732 TRAIN images,
        # one at a time, with no progress bar -- roughly 4x the TEST-set wait
        # above. Skip it with --skip-predicted-rule-based if you just want the
        # rest of the results quickly; it only affects one diagnostic baseline
        # row, not the primary SGSQ-Net numbers already printed above.
        if not args.skip_predicted_rule_based:
            print("  [predicted-IAR] running segmentation model over ~732 TRAIN images "
                  "to recalibrate the rule-based baseline (this is the slow step -- "
                  "several minutes, not a hang; pass --skip-predicted-rule-based to skip it)...")
            train_ds_for_rb = SeverityFeatureDataset(csv_path, seg_root, split="train", mask_source="predicted",
                                                      segmentation_model=seg_model, device=device, seg_img_size=seg_cfg["img_size"])
            print("  [predicted-IAR] TRAIN features done.")
            train_iar = np.array([train_ds_for_rb._features[i][0] for i in range(len(train_ds_for_rb))])
            test_iar = np.array([test_ds._features[i][0] for i in range(len(test_ds))])
            rb_preds = recalibrated_rule_based(train_iar, test_iar, num_classes)
            name_rb = "Rule-based predicted-IAR threshold (0 learned params, recalibrated on train)"
            preds_by_model[name_rb] = rb_preds
            report[name_rb] = qwk_report(name_rb, labels, rb_preds, num_classes, n_boot)
        else:
            print("  [skip] --skip-predicted-rule-based set -- skipping the predicted-IAR rule-based diagnostic.")
    else:
        print(f"[skip] segmentation-guided severity: need {sgsq_seg_ckpt.name}, {sgsq_sev_ckpt.name}, "
              f"and {norm_stats_path.name} all present under {ckpt_dir} -- found "
              f"seg={sgsq_seg_ckpt.exists()}, sev={sgsq_sev_ckpt.exists()}, norm={norm_stats_path.exists()}")

    # ---- (b) Box-derived-feature severity head (train_severity_real_interim.py) ----
    direct_ckpt = ckpt_dir / args.severity_direct_ckpt
    direct_norm_path = ckpt_dir / args.severity_direct_normalizer
    if direct_ckpt.exists() and direct_norm_path.exists():
        test_ds_direct = RealBoxFeatureDataset(csv_path, split="test")
        normalizer = FeatureNormalizer().load(direct_norm_path)

        model = SeverityHead(num_classes=num_classes, input_dim=2).to(device)
        model.load_state_dict(torch.load(direct_ckpt, map_location=device))
        model.eval()

        preds, labels = [], []
        with torch.no_grad():
            for i in range(len(test_ds_direct)):
                feat, label = test_ds_direct[i]
                feat = normalizer.transform(feat.unsqueeze(0).to(device))
                logits = model(feat)
                preds.append(SeverityHead.logits_to_class(logits).item())
                labels.append(label.item())
        preds = np.array(preds)
        labels = np.array(labels)
        if y_true_ref is None:
            y_true_ref = labels
        else:
            assert np.array_equal(labels, y_true_ref), (
                "row order / label mismatch between severity datasets on the TEST split -- "
                "check severity_labels.csv for split-order consistency before trusting comparisons"
            )
        name = "SGSQ-Net severity head (box-derived ITF/IAF features)"
        preds_by_model[name] = preds
        report[name] = qwk_report(name, labels, preds, num_classes, n_boot)

        # Recalibrated rule-based diagnostics on ITF and IAF directly (train-fit thresholds)
        for feat_name in ("inflamed_tooth_fraction", "inflamed_area_fraction"):
            df_all = pd.read_csv(csv_path)
            train_vals = df_all[df_all["split"] == "train"][feat_name].values
            test_vals = df_all[df_all["split"] == "test"][feat_name].values
            rb_preds = recalibrated_rule_based(train_vals, test_vals, num_classes)
            name_rb = f"Rule-based {feat_name} threshold (0 learned params, recalibrated on train)"
            preds_by_model[name_rb] = rb_preds
            report[name_rb] = qwk_report(name_rb, labels, rb_preds, num_classes, n_boot)
    else:
        print(f"[skip] box-derived-feature severity head: need {direct_ckpt.name} and {direct_norm_path.name} "
              f"under {ckpt_dir} -- found ckpt={direct_ckpt.exists()}, norm={direct_norm_path.exists()}")

    # ---- (b2) DETECTOR-DERIVED box-feature severity head (train_severity_detector_derived.py) ----
    # Same architecture as (b), but ITF/IAF come from the 7-class MGI detector's own
    # predictions (scripts/compute_detector_derived_features.py) instead of ground-truth
    # per-tooth annotations. This is the number a real deployed system (photo in, no
    # human-graded per-tooth input) would actually achieve -- compare it against (b),
    # the oracle, to see the "automation gap".
    detector_derived_ckpt = ckpt_dir / args.detector_derived_ckpt
    detector_derived_norm_path = ckpt_dir / args.detector_derived_normalizer
    detector_derived_csv = Path(cfg["paths"]["severity_dataset"]) / args.detector_derived_csv
    oracle_key = "SGSQ-Net severity head (box-derived ITF/IAF features)"
    if detector_derived_ckpt.exists() and detector_derived_norm_path.exists() and detector_derived_csv.exists():
        test_ds_det = RealBoxFeatureDataset(detector_derived_csv, split="test")
        normalizer = FeatureNormalizer().load(detector_derived_norm_path)

        model = SeverityHead(num_classes=num_classes, input_dim=2).to(device)
        model.load_state_dict(torch.load(detector_derived_ckpt, map_location=device))
        model.eval()

        preds, labels = [], []
        with torch.no_grad():
            for i in range(len(test_ds_det)):
                feat, label = test_ds_det[i]
                feat = normalizer.transform(feat.unsqueeze(0).to(device))
                logits = model(feat)
                preds.append(SeverityHead.logits_to_class(logits).item())
                labels.append(label.item())
        preds = np.array(preds)
        labels = np.array(labels)
        if y_true_ref is None:
            y_true_ref = labels
        else:
            assert np.array_equal(labels, y_true_ref), (
                "row order / label mismatch for the detector-derived severity dataset on TEST -- "
                "check severity_labels_detector_derived.csv row order against severity_labels.csv"
            )
        name = "SGSQ-Net severity head (DETECTOR-DERIVED ITF/IAF -- fully automated)"
        preds_by_model[name] = preds
        report[name] = qwk_report(name, labels, preds, num_classes, n_boot)

        if oracle_key in preds_by_model:
            gap_test = paired_qwk_bootstrap_test(labels, preds_by_model[oracle_key], preds, num_classes,
                                                  n_boot=max(n_boot, 5000))
            flag = "SIGNIFICANT gap (p<0.05)" if gap_test["p_value"] < 0.05 else "gap not significant"
            print(f"\n  [AUTOMATION GAP] oracle (ground-truth features) vs. detector-derived (fully automated):")
            print(f"    dQWK = {gap_test['diff']:+.4f}  [95% CI {gap_test['ci_low']:+.4f}, {gap_test['ci_high']:+.4f}]  "
                  f"p = {gap_test['p_value']:.4f}  -> {flag}")
            report["automation_gap_oracle_vs_detector_derived"] = gap_test
    else:
        print(f"[skip] detector-derived severity head: need {detector_derived_ckpt.name}, "
              f"{detector_derived_norm_path.name}, and {detector_derived_csv.name} -- found "
              f"ckpt={detector_derived_ckpt.exists()}, norm={detector_derived_norm_path.exists()}, "
              f"csv={detector_derived_csv.exists()}. Run scripts/train_detection_mgi.py -> "
              f"scripts/compute_detector_derived_features.py -> "
              f"scripts/train_severity_detector_derived.py first.")

    # ---- (b3) GROUND-TRUTH CONTINUOUS severity head (train_severity_groundtruth_continuous.py) ----
    # Apples-to-apples oracle: SAME continuous feature formula as (b2), but computed from
    # true per-tooth boxes instead of detector predictions. Comparing this against (b2)
    # isolates the effect of detector accuracy alone, since the feature formula is now
    # identical on both sides (the (b) vs (b2) comparison above conflates detector accuracy
    # with the binary-vs-continuous feature formula change).
    gt_cont_ckpt = ckpt_dir / args.gt_continuous_ckpt
    gt_cont_norm_path = ckpt_dir / args.gt_continuous_normalizer
    gt_cont_csv = Path(cfg["paths"]["severity_dataset"]) / args.gt_continuous_csv
    detector_derived_key = "SGSQ-Net severity head (DETECTOR-DERIVED ITF/IAF -- fully automated)"
    if gt_cont_ckpt.exists() and gt_cont_norm_path.exists() and gt_cont_csv.exists():
        test_ds_gtc = RealBoxFeatureDataset(gt_cont_csv, split="test")
        normalizer = FeatureNormalizer().load(gt_cont_norm_path)

        model = SeverityHead(num_classes=num_classes, input_dim=2).to(device)
        model.load_state_dict(torch.load(gt_cont_ckpt, map_location=device))
        model.eval()

        preds, labels = [], []
        with torch.no_grad():
            for i in range(len(test_ds_gtc)):
                feat, label = test_ds_gtc[i]
                feat = normalizer.transform(feat.unsqueeze(0).to(device))
                logits = model(feat)
                preds.append(SeverityHead.logits_to_class(logits).item())
                labels.append(label.item())
        preds = np.array(preds)
        labels = np.array(labels)
        if y_true_ref is None:
            y_true_ref = labels
        else:
            assert np.array_equal(labels, y_true_ref), (
                "row order / label mismatch for the ground-truth-continuous severity dataset on TEST -- "
                "check severity_labels_groundtruth_continuous.csv row order against severity_labels.csv"
            )
        name = "SGSQ-Net severity head (ground-truth CONTINUOUS ITF/IAF -- apples-to-apples oracle)"
        preds_by_model[name] = preds
        report[name] = qwk_report(name, labels, preds, num_classes, n_boot)

        if detector_derived_key in preds_by_model:
            gap_test = paired_qwk_bootstrap_test(labels, preds, preds_by_model[detector_derived_key],
                                                  num_classes, n_boot=max(n_boot, 5000))
            flag = "SIGNIFICANT gap (p<0.05)" if gap_test["p_value"] < 0.05 else "gap not significant"
            print(f"\n  [APPLES-TO-APPLES AUTOMATION GAP] ground-truth-continuous oracle vs. "
                  f"detector-derived (SAME feature formula, only the source -- true boxes vs. "
                  f"detector predictions -- differs):")
            print(f"    dQWK = {gap_test['diff']:+.4f}  [95% CI {gap_test['ci_low']:+.4f}, {gap_test['ci_high']:+.4f}]  "
                  f"p = {gap_test['p_value']:.4f}  -> {flag}")
            report["automation_gap_gt_continuous_vs_detector_derived"] = gap_test
    else:
        print(f"[skip] ground-truth-continuous severity head: need {gt_cont_ckpt.name}, "
              f"{gt_cont_norm_path.name}, and {gt_cont_csv.name} -- found "
              f"ckpt={gt_cont_ckpt.exists()}, norm={gt_cont_norm_path.exists()}, "
              f"csv={gt_cont_csv.exists()}. Run scripts/compute_ground_truth_continuous_features.py -> "
              f"scripts/train_severity_groundtruth_continuous.py first.")

    # ---- (c) End-to-end ablation baselines (EfficientNet-B0, ViT-B/16) ----
    for model_key, model_cls, ckpt_name, display_name in [
        ("efficientnet", EfficientNetB0Classifier, args.efficientnet_ckpt, "EfficientNet-B0 (end-to-end ablation)"),
        ("vit", ViTClassifier, args.vit_ckpt, "ViT-B/16 (end-to-end ablation)"),
    ]:
        ckpt_path = ckpt_dir / "baselines" / ckpt_name
        if not ckpt_path.exists():
            print(f"[skip] {display_name}: checkpoint not found at {ckpt_path}")
            continue
        print(f"  [{display_name}] running inference on 182 test images (batched, should be quick)...")
        model = model_cls(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        ds = RawImageSeverityDataset(csv_path, images_root, split="test", img_size=224, train=False)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        preds, labels = [], []
        with torch.no_grad():
            for images, lbls in loader:
                images = images.to(device)
                outputs = model(images)
                preds.extend(outputs.argmax(dim=1).cpu().tolist())
                labels.extend(lbls.tolist())
        preds = np.array(preds)
        labels = np.array(labels)
        if y_true_ref is None:
            y_true_ref = labels
        else:
            assert np.array_equal(labels, y_true_ref), f"label order mismatch for {display_name} on TEST split"
        preds_by_model[display_name] = preds
        report[display_name] = qwk_report(display_name, labels, preds, num_classes, n_boot)

    # ---- Paired significance vs. the primary SGSQ-Net approach ----
    # DETECTOR-DERIVED is listed first: it is the only fully-automated,
    # non-circular severity result (photo in, no ground-truth per-tooth
    # labels used at inference). The ground-truth-continuous "oracle" is
    # deliberately NOT a candidate here -- confirmed circular (Spearman
    # rho=0.978 between its own feature and severity_label, i.e. severity_label
    # was constructed by quantile-binning ~this same quantity), so comparing
    # against it as a performance ceiling would be misleading; it stays in
    # preds_by_model only as a reported sanity-check row, not a baseline any
    # other method is benchmarked against here.
    primary_candidates = [
        "SGSQ-Net severity head (DETECTOR-DERIVED ITF/IAF -- fully automated)",
        "SGSQ-Net (segmentation-guided severity)",
        "SGSQ-Net severity head (box-derived ITF/IAF features)",
    ]
    primary = next((c for c in primary_candidates if c in preds_by_model), None)

    # Excluded from the significance loop entirely: comparing other methods against
    # this "oracle" would misleadingly suggest they're deficient relative to a ceiling,
    # when the ceiling itself is an artifact of severity_label's own construction.
    circular_oracle_key = "SGSQ-Net severity head (ground-truth CONTINUOUS ITF/IAF -- apples-to-apples oracle)"

    if primary is not None and y_true_ref is not None:
        print("\n" + "-" * 70)
        print(f"Paired permutation test (with bootstrap CI of the difference): '{primary}' vs. each other model (QWK)")
        print("-" * 70)
        if circular_oracle_key in preds_by_model:
            print(f"  [note] '{circular_oracle_key}' is excluded from this comparison -- confirmed "
                  f"circular (severity_label was constructed by quantile-binning essentially this "
                  f"same ground-truth feature; see diagnose_detector_features.py / Spearman check). "
                  f"Its QWK is reported above as a label-construction sanity check only, never as a "
                  f"performance ceiling other methods are benchmarked against.")
        sig_rows = []
        for name, preds in preds_by_model.items():
            if name == primary or name == circular_oracle_key:
                continue
            test = paired_qwk_bootstrap_test(y_true_ref, preds_by_model[primary], preds, num_classes,
                                              n_boot=max(n_boot, 5000))
            perm = paired_qwk_permutation_test(y_true_ref, preds_by_model[primary], preds, num_classes)
            sig_rows.append({"comparison": f"{primary} vs {name}", "model": name,
                             "diff": perm["diff"], "ci_low": test["ci_low"], "ci_high": test["ci_high"],
                             "p_perm": perm["p_perm"]})
        if sig_rows:
            adj = holm_adjust([r["p_perm"] for r in sig_rows])
            for r, ph in zip(sig_rows, adj):
                r["p_holm"] = float(ph)
                print(f"  vs. {r['model']}: dQWK = {r['diff']:+.4f}  [95% CI {r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
                      f"p_perm = {r['p_perm']:.4f}  p_Holm = {r['p_holm']:.4f}")
            print("  (paired permutation test, 20,000 swaps; smallest possible p = 0.00005)")
            pd.DataFrame(sig_rows).to_csv(ckpt_dir / "test_significance_severity.csv", index=False)

        print(f"\nFinal classification report -- PRIMARY approach ('{primary}'), HELD-OUT TEST split:")
        print(classification_report(y_true_ref, preds_by_model[primary], labels=list(range(num_classes)),
                                     target_names=class_names, zero_division=0))
    else:
        print("\n[note] no primary SGSQ-Net severity checkpoint available -- no paired significance computed.")

    # ---- Confusion matrices (raw counts) for every evaluated severity method ----
    # Filled a real gap flagged during paper review: classification_report gives derived
    # precision/recall/F1, but not the raw confusion matrix a reviewer or reader would want
    # to see directly (e.g. to check whether errors are off-by-one on the ordinal scale, as
    # expected for a reasonable model, vs. scattered arbitrarily). Printed for every method,
    # not just the primary, so the segmentation-guided negative finding's near-constant-class
    # collapse is visible as a matrix (not only as near-zero per-class recall) alongside the
    # primary method's more spread, mostly-near-diagonal pattern.
    if y_true_ref is not None and preds_by_model:
        print("\n" + "-" * 70)
        print("Confusion matrices (rows = true grade, columns = predicted grade), HELD-OUT TEST")
        print("-" * 70)
        cm_report = {}
        for name, preds in preds_by_model.items():
            cm = confusion_matrix(y_true_ref, preds, labels=list(range(num_classes)))
            print(f"\n{name}:")
            header = "        " + "".join(f"pred_{c+1:<6}" for c in range(num_classes))
            print(header)
            for i, row in enumerate(cm):
                print(f"true_{i+1:<3}" + "".join(f"{v:<11d}" for v in row))
            cm_report[name] = cm.tolist()
        report["confusion_matrices"] = cm_report

    return report


# ==========================================================================
# Main
# ==========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--sgsq-seg-ckpt", default="segmentation_best.pt")
    ap.add_argument("--deeplab-ckpt", default="deeplabv3_best.pt")
    ap.add_argument("--sgsq-sev-ckpt", default="severity_head_best.pt")
    ap.add_argument("--sgsq-sev-normalizer", default="severity_feature_stats.json")
    ap.add_argument("--severity-direct-ckpt", default="severity_head_real_interim_best.pt")
    ap.add_argument("--severity-direct-normalizer", default="severity_feature_stats_real_interim.json")
    ap.add_argument("--detector-derived-ckpt", default="severity_head_detector_derived_best.pt")
    ap.add_argument("--detector-derived-normalizer", default="severity_feature_stats_detector_derived.json")
    ap.add_argument("--detector-derived-csv", default="severity_labels_detector_derived.csv",
                     help="Filename under <severity_dataset>/ (not a full path) written by "
                          "scripts/compute_detector_derived_features.py")
    ap.add_argument("--gt-continuous-ckpt", default="severity_head_gt_continuous_best.pt")
    ap.add_argument("--gt-continuous-normalizer", default="severity_feature_stats_gt_continuous.json")
    ap.add_argument("--gt-continuous-csv", default="severity_labels_groundtruth_continuous.csv",
                     help="Filename under <severity_dataset>/ (not a full path) written by "
                          "scripts/compute_ground_truth_continuous_features.py -- the apples-to-apples "
                          "oracle (same continuous feature formula as detector-derived, but from "
                          "ground-truth per-tooth boxes instead of detector predictions)")
    ap.add_argument("--efficientnet-ckpt", default="efficientnet_classifier_best.pt")
    ap.add_argument("--vit-ckpt", default="vit_classifier_best.pt")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-predicted-rule-based", action="store_true",
                     help="Skip recalibrating the predicted-IAR rule-based baseline on the train split "
                          "(the slowest step -- reruns the segmentation model over ~732 train images "
                          "one at a time with no progress bar). Use this for a fast first look at results.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    print(f"Device: {device}")

    full_report = {}
    full_report["segmentation"] = run_segmentation_eval(
        cfg, ckpt_dir, device, args.n_boot, args.sgsq_seg_ckpt, args.deeplab_ckpt)
    full_report["severity"] = run_severity_eval(cfg, ckpt_dir, device, args.n_boot, args)

    with open(ckpt_dir / "test_set_full_report.json", "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\nSaved: {ckpt_dir / 'test_set_full_report.json'}")
    print(f"Saved: {ckpt_dir / 'test_significance_severity.csv'} (if a primary severity checkpoint was available)")
    print("\nThese TEST-split numbers are the ones that belong in the paper's Results section.")
    print("Validation-split numbers (from earlier training logs / benchmark_compare.py) should only be")
    print("cited, if at all, in a methods note about early stopping -- never as headline results.")


if __name__ == "__main__":
    main()
