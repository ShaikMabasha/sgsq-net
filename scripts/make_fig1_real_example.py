"""
scripts/make_fig1_real_example.py
=================================
Builds the paper's task-overview figure (input -> detection -> severity) from a REAL
held-out TEST image and the REAL trained models -- no hand-drawn boxes or grades.

  (a) the intraoral photograph (cropped to the detected mouth region)
  (b) the 7-class MGI detector's own boxes (conf >= 0.25) with per-tooth MGI grade
  (c) the image-level severity (S1-S5) predicted by the primary (detector-derived) head,
      the ground-truth severity, and the ITF / IAF features that produced it

Image choice (to avoid cherry-picking): by default a TEST image is drawn at random with a
fixed --seed; the file name / ground truth / prediction are printed so you can state the
selection rule in the caption. Use --image <filename> to force a specific test image.

Usage (from the sgsq_pipeline root):
    python3 scripts/make_fig1_real_example.py --config configs/config.yaml
    python3 scripts/make_fig1_real_example.py --config configs/config.yaml --seed 7
    python3 scripts/make_fig1_real_example.py --config configs/config.yaml --image IMG_0123.jpg

Writes: figures/fig1_task_overview_real.pdf / .png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch

TOOTH_CLASSES = {2, 3, 4, 5, 6}   # per-tooth MGI classes in the 7-class scheme
MGI_OFFSET = 2                    # mgi grade = class id - 2  (0..4)
CONF = 0.25                       # same threshold used to build the ITF/IAF features

# One sequential ramp (light -> dark = low -> high inflammation), used in BOTH panels.
RAMP = ["#ffe9b0", "#fdc16a", "#f48c3f", "#d9532b", "#9c1d1d"]
GINGIVA_COL = "#1b8a8f"


def itf_iaf(cls, xywhn):
    """Same formula as compute_detector_derived_features_v2.py."""
    levels = (cls - MGI_OFFSET).astype(float)
    areas = xywhn[:, 2] * xywhn[:, 3]
    itf = float(levels.mean() / 4.0)
    iaf = float((areas * levels).sum() / areas.sum() / 4.0) if areas.sum() > 0 else 0.0
    return itf, iaf


def crop_window(xyxy, W, H, mx=0.10, my=0.30):
    """Crop to the union of all detected boxes plus a margin (also trims the face)."""
    x0, y0 = xyxy[:, 0].min(), xyxy[:, 1].min()
    x1, y1 = xyxy[:, 2].max(), xyxy[:, 3].max()
    dx, dy = (x1 - x0) * mx, (y1 - y0) * my
    return (max(0, x0 - dx), min(W, x1 + dx), max(0, y0 - dy), min(H, y1 + dy))


def draw_figure(img, cls, xyxy, pred_grade, gt_grade, itf, iaf, out):
    """img: PIL RGB image; cls/xyxy: detector outputs (numpy); grades are 0-based (S1 = 0)."""
    W, H = img.size
    cx0, cx1, cy0, cy1 = crop_window(xyxy, W, H)
    aspect = (cx1 - cx0) / (cy1 - cy0)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9),
                             gridspec_kw={"width_ratios": [aspect, aspect, 0.95]})
    for ax in axes:
        ax.axis("off")
    for ax in axes[:2]:
        ax.imshow(img)
        ax.set_xlim(cx0, cx1)
        ax.set_ylim(cy1, cy0)

    for c, (x1, y1, x2, y2) in zip(cls, xyxy):
        is_tooth = c in TOOTH_CLASSES
        col = RAMP[c - MGI_OFFSET] if is_tooth else GINGIVA_COL
        axes[1].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=col,
                                    lw=1.3 if is_tooth else 1.1, ls="-" if is_tooth else "--"))
        if is_tooth:
            axes[1].text(x1 + 2, y1 + 2, str(c - MGI_OFFSET), fontsize=6, va="top", ha="left",
                         weight="bold", color="black",
                         bbox=dict(boxstyle="square,pad=0.12", fc=col, ec="none"))
    axes[1].legend(handles=[Patch(fc=RAMP[g], ec="none", label=f"MGI {g}") for g in range(5)],
                   loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=5, frameon=False,
                   fontsize=6, handlelength=1.0, columnspacing=0.9, handletextpad=0.3)

    # ---- panel (c) ----
    ax = axes[2]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    xs = [0.115 + i * 0.19 for i in range(5)]
    for i in range(5):
        hi = (i == pred_grade)
        ax.add_patch(Rectangle((0.03 + i * 0.19, 0.46), 0.17, 0.17, fc=RAMP[i], ec="black",
                               lw=2.2 if hi else 0.6))
        ax.text(xs[i], 0.545, f"S{i + 1}", ha="center", va="center", fontsize=7.5,
                weight="bold" if hi else "normal")
    arrow = dict(arrowstyle="-|>", lw=0.9, color="#222222", shrinkA=0, shrinkB=1)
    if pred_grade == gt_grade:
        ax.annotate("predicted = true", xy=(xs[pred_grade], 0.64), xytext=(xs[pred_grade], 0.80),
                    ha="center", va="bottom", fontsize=6.5, arrowprops=arrow)
    else:
        ax.annotate("predicted", xy=(xs[pred_grade], 0.64), xytext=(xs[pred_grade], 0.80),
                    ha="center", va="bottom", fontsize=6.5, arrowprops=arrow)
        ax.annotate("true", xy=(xs[gt_grade], 0.45), xytext=(xs[gt_grade], 0.29),
                    ha="center", va="top", fontsize=6.5, arrowprops=arrow)
    ax.text(0.5, 0.97, "image-level severity S1–S5 (low → high)", ha="center", va="top",
            fontsize=6.5, style="italic")
    ax.text(0.5, 0.06, f"ITF = {itf:.2f}   IAF = {iaf:.2f}", ha="center", va="bottom", fontsize=7)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.10, wspace=0.14)
    fig.canvas.draw()                      # resolves the aspect-constrained axes positions
    pa, pb, pc = (a.get_position() for a in axes)
    axes[2].set_position([pc.x0, pa.y0, pc.width, pa.height])   # (c) as tall as the photos
    top = pa.y1 + 0.02
    for ax_, t in zip(axes, ["(a) Input", "(b) Detection", "(c) Severity"]):
        p = ax_.get_position()
        fig.text((p.x0 + p.x1) / 2, top, t, ha="center", va="bottom", fontsize=9)
    mid = (pa.y0 + pa.y1) / 2
    for a, b in [(0, 1), (1, 2)]:
        p0, p1 = axes[a].get_position(), axes[b].get_position()
        fig.add_artist(FancyArrowPatch((p0.x1 + 0.003, mid), (p1.x0 - 0.003, mid),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=10, lw=1.3, color="#333333"))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", dpi=300)
    fig.savefig(str(out) + ".png", dpi=300)
    plt.close(fig)
    print(f"Saved {out}.pdf / .png")


def main():
    import pandas as pd
    import torch
    import yaml
    from PIL import Image
    from ultralytics import YOLO

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from models.severity_head import SeverityHead, FeatureNormalizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--image", default=None, help="TEST-split file name to use (default: random with --seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default="severity_labels_detector_derived_v2.csv")
    ap.add_argument("--detector-ckpt", default="detection_mgi_best.pt")
    ap.add_argument("--sev-ckpt", default="severity_head_detector_derived_best.pt")
    ap.add_argument("--sev-norm", default="severity_feature_stats_detector_derived.json")
    ap.add_argument("--out", default="figures/fig1_task_overview_real")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    img_dir = Path(cfg["paths"]["segmentation_dataset"]) / "images"
    num_classes = len(cfg["severity"]["classes"])

    df = pd.read_csv(Path(cfg["paths"]["severity_dataset"]) / args.csv)
    test = df[df["split"] == "test"].reset_index(drop=True)
    if args.image:
        row = test[test["filename"] == args.image]
        if row.empty:
            raise SystemExit(f"{args.image} is not in the TEST split of {args.csv}")
        row = row.iloc[0]
    else:
        row = test.iloc[int(np.random.default_rng(args.seed).integers(len(test)))]
    fname, gt_grade = row["filename"], int(row["severity_label"])

    # ---- detection ----
    det = YOLO(str(ckpt_dir / args.detector_ckpt))
    res = det.predict(str(img_dir / fname), verbose=False, conf=CONF)[0]
    boxes = res.boxes
    cls = boxes.cls.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy()
    xywhn = boxes.xywhn.cpu().numpy()
    tooth = np.isin(cls, list(TOOTH_CLASSES))
    if tooth.sum() == 0:
        raise SystemExit("No teeth detected in this image -- pick another with --seed or --image.")
    itf, iaf = itf_iaf(cls[tooth], xywhn[tooth])

    # ---- severity head (primary, detector-derived) ----
    normalizer = FeatureNormalizer().load(ckpt_dir / args.sev_norm)
    head = SeverityHead(num_classes=num_classes, input_dim=2).to(device)
    head.load_state_dict(torch.load(ckpt_dir / args.sev_ckpt, map_location=device))
    head.eval()
    with torch.no_grad():
        feat = normalizer.transform(torch.tensor([[itf, iaf]], dtype=torch.float32).to(device))
        pred_grade = int(SeverityHead.logits_to_class(head(feat)).item())

    print(f"Selected TEST image  : {fname}")
    print(f"Ground-truth severity: S{gt_grade + 1}   Predicted severity: S{pred_grade + 1}")
    print(f"ITF={itf:.3f}  IAF={iaf:.3f}  teeth detected={int(tooth.sum())}")
    print("Selection rule for the caption:",
          f"--image {args.image}" if args.image else f"random TEST image, seed={args.seed}")

    img = Image.open(img_dir / fname).convert("RGB")
    draw_figure(img, cls, xyxy, pred_grade, gt_grade, itf, iaf, args.out)


if __name__ == "__main__":
    main()
