"""
scripts/make_qualitative_errors.py
==================================
Qualitative examples from the REAL held-out TEST split, produced by the real primary model
(MGI detector -> ITF/IAF -> CORAL head). Nothing is hand-picked: examples are drawn at random (--seed)
from rule-defined groups.

  Row 1 (correct):   S1, S3, S5 predicted correctly
  Row 2 (errors):    true S2 under-estimated (pred S1), true S2 over-estimated (pred S3), and the
                     largest over-estimate of a true S2 (pred S4/S5) -- falling back to any error with
                     the largest grade distance if a group is empty.
Each tile shows the cropped photograph with the detector's per-tooth MGI boxes (conf >= 0.25), the true and
predicted grade, and the ITF / IAF values that produced the prediction. Prints the file names so the caption
can state the selection rule.

Usage (from the sgsq_pipeline root):
    python3 scripts/make_qualitative_errors.py --config configs/config.yaml
    python3 scripts/make_qualitative_errors.py --config configs/config.yaml --seed 3
Writes figures/fig_qualitative_examples.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Patch

sys.path.append(str(Path(__file__).resolve().parent))
from make_fig1_real_example import itf_iaf, RAMP, GINGIVA_COL, TOOTH_CLASSES, MGI_OFFSET, CONF  # noqa: E402

CELL_ASPECT = 1.6


def crop(xyxy, W, H, mx=0.08, my=0.25):
    x0, y0, x1, y1 = xyxy[:, 0].min(), xyxy[:, 1].min(), xyxy[:, 2].max(), xyxy[:, 3].max()
    dx, dy = (x1 - x0) * mx, (y1 - y0) * my
    x0, x1, y0, y1 = max(0, x0 - dx), min(W, x1 + dx), max(0, y0 - dy), min(H, y1 + dy)
    w, h = x1 - x0, y1 - y0
    if w / h < CELL_ASPECT:
        nw = min(W, h * CELL_ASPECT); c = (x0 + x1) / 2; x0, x1 = max(0, c - nw / 2), min(W, c + nw / 2)
    else:
        nh = min(H, w / CELL_ASPECT); c = (y0 + y1) / 2; y0, y1 = max(0, c - nh / 2), min(H, c + nh / 2)
    return x0, x1, y0, y1


def pick(rng, items, cond, used):
    pool = [i for i in items if cond(i) and i["name"] not in used]
    return pool[int(rng.integers(len(pool)))] if pool else None


def select(items, seed):
    rng = np.random.default_rng(seed)
    used, out = set(), []
    plan = [("correct", lambda i: i["true"] == i["pred"] == 0), ("correct", lambda i: i["true"] == i["pred"] == 2),
            ("correct", lambda i: i["true"] == i["pred"] == 4),
            ("error", lambda i: i["true"] == 1 and i["pred"] == 0), ("error", lambda i: i["true"] == 1 and i["pred"] == 2),
            ("error", lambda i: i["true"] == 1 and i["pred"] >= 3)]
    for kind, cond in plan:
        it = pick(rng, items, cond, used)
        if it is None and kind == "error":                      # fallback: biggest remaining error
            errs = sorted([i for i in items if i["true"] != i["pred"] and i["name"] not in used],
                          key=lambda i: -abs(i["true"] - i["pred"]))
            it = errs[0] if errs else None
        if it is None:                                          # fallback: any unused item
            it = pick(rng, items, lambda i: True, used)
        used.add(it["name"]); out.append(it)
    return out


def draw_grid(chosen, out):
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.1))
    for ax, it in zip(axes.ravel(), chosen):
        img, cls, xyxy = it["img"], it["cls"], it["xyxy"]
        W, H = img.size
        x0, x1, y0, y1 = crop(xyxy, W, H)
        ax.imshow(img); ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
        for c, (a, b, c2, d2) in zip(cls, xyxy):
            tooth = c in TOOTH_CLASSES
            col = RAMP[c - MGI_OFFSET] if tooth else GINGIVA_COL
            ax.add_patch(Rectangle((a, b), c2 - a, d2 - b, fill=False, ec=col, lw=1.0 if tooth else 0.8,
                                   ls="-" if tooth else "--"))
            if tooth:
                ax.text(a + 2, b + 2, str(c - MGI_OFFSET), fontsize=5, va="top", ha="left", weight="bold",
                        bbox=dict(boxstyle="square,pad=0.1", fc=col, ec="none"))
        ok = it["true"] == it["pred"]
        ax.set_title(f"true S{it['true'] + 1} → predicted S{it['pred'] + 1}", fontsize=7.5, pad=2,
                     color="#1f1f1f", weight="bold" if not ok else "normal")
        ax.text(0.5, -0.03, f"ITF {it['itf']:.2f}  IAF {it['iaf']:.2f}", transform=ax.transAxes, ha="center",
                va="top", fontsize=6.5, color="#4a4a4a")
    fig.text(0.008, 0.73, "correct", rotation=90, va="center", fontsize=8)
    fig.text(0.008, 0.28, "errors", rotation=90, va="center", fontsize=8)
    handles = [Patch(fc=RAMP[k], ec="none", label=f"MGI {k}") for k in range(5)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=6.5, handlelength=1.1,
               columnspacing=1.0, handletextpad=0.3)
    fig.subplots_adjust(left=0.03, right=0.995, top=0.94, bottom=0.09, wspace=0.03, hspace=0.34)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300); plt.close(fig); print("Saved", out)


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default="severity_labels_detector_derived_v2.csv")
    ap.add_argument("--detector-ckpt", default="detection_mgi_best.pt")
    ap.add_argument("--sev-ckpt", default="severity_head_detector_derived_best.pt")
    ap.add_argument("--sev-norm", default="severity_feature_stats_detector_derived.json")
    ap.add_argument("--out", default="figures/fig_qualitative_examples.png")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    ckpt = Path(cfg["paths"]["checkpoints"])
    img_dir = Path(cfg["paths"]["segmentation_dataset"]) / "images"
    n_cls = len(cfg["severity"]["classes"])
    df = pd.read_csv(Path(cfg["paths"]["severity_dataset"]) / a.csv)
    test = df[df["split"] == "test"].reset_index(drop=True)

    det = YOLO(str(ckpt / a.detector_ckpt))
    norm = FeatureNormalizer().load(ckpt / a.sev_norm)
    head = SeverityHead(num_classes=n_cls, input_dim=2).to(device)
    head.load_state_dict(torch.load(ckpt / a.sev_ckpt, map_location=device)); head.eval()

    items = []
    for _, r in test.iterrows():
        res = det.predict(str(img_dir / r["filename"]), verbose=False, conf=CONF)[0]
        cls = res.boxes.cls.cpu().numpy().astype(int)
        xyxy = res.boxes.xyxy.cpu().numpy(); xywhn = res.boxes.xywhn.cpu().numpy()
        t = np.isin(cls, list(TOOTH_CLASSES))
        if t.sum() == 0:
            continue
        itf, iaf = itf_iaf(cls[t], xywhn[t])
        with torch.no_grad():
            f = norm.transform(torch.tensor([[itf, iaf]], dtype=torch.float32).to(device))
            pred = int(SeverityHead.logits_to_class(head(f)).item())
        items.append(dict(name=r["filename"], true=int(r["severity_label"]), pred=pred, itf=itf, iaf=iaf,
                          cls=cls, xyxy=xyxy))
    print(f"{len(items)} test images scored; accuracy = {np.mean([i['true'] == i['pred'] for i in items]):.3f}")
    chosen = select(items, a.seed)
    for it in chosen:
        print(f"{it['name']}: true S{it['true'] + 1}, predicted S{it['pred'] + 1}")
        it["img"] = Image.open(img_dir / it["name"]).convert("RGB")
    print("Selection rule for caption: random example per group, seed =", a.seed)
    draw_grid(chosen, a.out)


if __name__ == "__main__":
    main()
