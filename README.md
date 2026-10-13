# SGSQ-Net: Quantifying Gingival Inflammation Severity from Intraoral Photographs

Code for estimating gingival inflammation severity (five relative groups, S1–S5) from a single intraoral photograph.
A seven-class, Modified Gingival Index (MGI)-aware YOLOv8n detector grades each tooth; the detections are summarized as two normalized mean-grade scores (ITF and IAF); and a small CORAL ordinal head maps them to S1–S5. A segmentation-guided alternative (ResNet34 U-Net) is included and evaluated head to head.

> **Paper:** *SGSQ-Net: Quantifying Gingival Inflammation Severity from Intraoral Photographs with MGI-Aware Tooth Detection and Ordinal Regression* — <journal, status, DOI>
> **Authors:** Mabasha Shaik, Sumit Gupta, Kancharagunta Kishan Babu

**This is a research baseline, not a clinical tool.** S1–S5 are dataset-relative quantile groups, not clinical stages, and the model has not been externally or prospectively validated.

---

## 1. Method at a glance

| Stage | What it does |
|---|---|
| Detector | YOLOv8n, 7 classes: `upper_gingiva`, `lower_gingiva`, and five per-tooth classes `tooth_mgi0 … tooth_mgi4` (class id = MGI + 2). Predictions kept at confidence ≥ 0.25 (fixed, not tuned). |
| Features | For the *n* tooth detections with predicted MGI grade *c<sub>i</sub>* and normalized box area *a<sub>i</sub>*: **ITF** = (1/4n)·Σ c<sub>i</sub> (normalized mean grade) and **IAF** = ¼·Σ a<sub>i</sub>c<sub>i</sub> / Σ a<sub>i</sub> (area-weighted). The names are historical; neither is a share of teeth/area with MGI ≥ 1. Features are z-scored with training-split statistics. |
| Head | CORAL ordinal head: MLP 2→16→16 (ReLU) → one shared score → 4 learned thresholds; 340 parameters. BCE on cumulative targets, Adam (lr 1e-3, batch 16), early stopping on validation QWK (patience 15). |
| Alternative | ResNet34 U-Net (256×256) trained on box-filled masks of teeth with MGI ≥ 1 → inflamed-area ratio (IAR) + mean redness → separately trained identical CORAL head. |

**Labels.** S1–S5 are five quantile bins of the image's mean MGI (edges fit on the training split). S1 is the lowest-inflammation group in this cohort, not a healthy group. Tooth-level MGI (0–4) and image-level severity (S1–S5) are different scales.

---

## 2. Data

We use the public intraoral gingivitis dataset of Duy et al., *Data in Brief* 57 (2024) 110960 (Mendeley Data): 1,096 RGB photographs with per-tooth bounding boxes and MGI grades.
**The images are not redistributed here.** Download them from the original record and check its licence terms before use: <Mendeley Data URL>.

- Split: the dataset's published 732 / 182 / 182 (train / val / test) split, used unchanged. It is not stratified, and no patient identifiers are available, so images are treated as independent and patient overlap cannot be checked.
- `splits/split_labels.csv`: `filename, split, severity_label` for every image.
- Place the data under `sgsq_data/` as expected by `configs/config.yaml`.

---

## 3. Installation

```bash
git clone https://github.com/<your-username>/sgsq-net.git
cd sgsq-net
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Tested with: Python <x.y>, PyTorch <x.y>, Ultralytics <x.y>, CUDA <x.y>. A GPU is recommended.

---

## 4. Reproducing the results

Everything uses seed 42. Adjust the script arguments to your paths (see `--help` of each script).

```bash
# 0. Convert the downloaded dataset to the project format
python data/convert_mendeley_dataset.py ...

# 1. Train the 7-class MGI-aware detector
python scripts/train_detection_mgi.py --config configs/config.yaml

# 2. Compute detector-derived ITF/IAF features for all images (conf >= 0.25)
python scripts/compute_detector_derived_features_v2.py --config configs/config.yaml

# 3. Train the CORAL severity head on the detector-derived features
python scripts/train_severity_detector_derived.py --config configs/config.yaml

# 4. (Optional) Segmentation-guided pathway and baselines
python scripts/train_segmentation.py --config configs/config.yaml
python scripts/train_baseline_classifier.py ...   # EfficientNet-B0, ViT-B/16

# 5. Held-out test evaluation (paired permutation test + Holm correction)
python scripts/evaluate_test_set.py --config configs/config.yaml \
    --detector-derived-csv severity_labels_detector_derived_v2.csv
```

**Important:** the detector-derived CSV must be the v2 file (continuous features). The older file contains binary features equal to 1.0 for every image, and a head evaluated on it collapses to a constant prediction (QWK 0.000).

Analysis scripts used for the paper's tables and figures: `scripts/ablate_itf_iaf_features.py` (ITF/IAF ablation), `scripts/diagnose_feature_shift.py` (feature shift between splits), `scripts/check_detector_perclass.py` (per-class detector metrics), and the `make_*_figures.py` scripts.

---

## 5. Main results (held-out test set, n = 182)

Quadratic weighted kappa (QWK) with 95% bootstrap CI (B = 2,000). *p*: two-sided paired permutation test (20,000 swaps) against the primary model, before Holm correction.

| Approach | Automated | QWK [95% CI] | p |
|---|---|---|---|
| **SGSQ-Net, detector-derived ITF/IAF (primary)** | Yes | **0.467 [0.329, 0.588]** | — |
| EfficientNet-B0 | Yes | 0.301 [0.191, 0.404] | 0.037 |
| Box-derived ITF/IAF head (binary, ground-truth boxes) | No | 0.259 [0.157, 0.355] | 0.020 |
| ViT-B/16 | Yes | 0.247 [0.101, 0.382] | 0.007 |
| Rule-based threshold, ITF/IAF (ground-truth boxes) | No | 0.119 [0.067, 0.178] | < 0.001 |
| Rule-based threshold, predicted IAR | Yes | −0.102 [−0.174, −0.032] | < 0.001 |
| SGSQ-Net, segmentation-guided | Yes | −0.111 [−0.195, −0.027] | < 0.001 |

All comparisons remain significant after Holm correction (largest adjusted p ≈ 0.040). A head trained on the ground-truth continuous aggregate reaches QWK 0.966, but this is circular by construction (the labels are quantile bins of the same quantity) and is excluded from all tests.

Other test-set numbers: accuracy 0.297, macro-F1 0.295 (majority-class rate 0.440); 76.4% of predictions within one grade; the primary model predicts S1 for 98 of 182 images. U-Net Dice 0.872 [0.862, 0.881] vs DeepLabV3 0.871 [0.862, 0.880] against box-derived masks. Detector (all classes) precision / recall / mAP@50 / mAP@50:95 = 0.507 / 0.723 / 0.544 / 0.352; detector precision and recall are reported at the F1-maximizing confidence on the evaluated split and are descriptive.

---

## 6. Limitations

- One public dataset, one non-stratified split without patient identifiers, one training seed (a second run of the same head moved test QWK by about 0.05), and untuned baselines.
- S1–S5 are relative quantile groups, not clinical grades; agreement is moderate and the lowest grade is over-predicted.
- The segmentation target is box-derived (tooth plus adjacent gingiva), so Dice measures agreement with coarse regions.
- The head is trained on features of images the detector was itself fitted on; features are less separated on unseen images (see `scripts/diagnose_feature_shift.py`).
- No external or prospective validation; no calibration of the severity probabilities.

---

## 7. Repository structure

```
configs/            config.yaml (paths, hyperparameters)
data/               dataset conversion and dataset classes
models/             U-Net, CORAL severity head, baselines
scripts/            training, feature extraction, evaluation, analysis, figures
splits/             split_labels.csv (filename, split, severity_label)
requirements.txt
```

## 8. Pretrained weights

Weights behind the paper's numbers (detector, detector-derived severity head with its normalizer, segmentation model, baselines): <GitHub Release or Zenodo DOI>.

## 9. Citation

```bibtex
@article{shaik2026sgsqnet,
  title   = {SGSQ-Net: Quantifying Gingival Inflammation Severity from Intraoral Photographs with MGI-Aware Tooth Detection and Ordinal Regression},
  author  = {Shaik, Mabasha and Gupta, Sumit and Babu, Kancharagunta Kishan},
  journal = {<journal>},
  year    = {<year>},
  doi     = {<doi>}
}
```
Please also cite the dataset: Duy et al., *A dental intraoral image dataset of gingivitis for image captioning*, Data in Brief 57 (2024) 110960, doi:10.1016/j.dib.2024.110960.

## 10. Licence and contact

Code: <MIT or your chosen licence>. The dataset is subject to its own licence.
Contact: <corresponding author email>
