"""
scripts/diagnose_feature_shift.py
=================================
CPU-only diagnostic for one hypothesis in the paper: the severity head is fit on detector features computed
for TRAIN images (which the detector was trained on), but applied to unseen TEST images. If the detector's
features differ between seen and unseen images, the head's thresholds are misplaced on test.

For every severity grade (S1-S5) and every split (train / val / test) it shows the distribution of the
detector-derived ITF (and IAF), and prints the median and the Spearman correlation with the grade per split.
Reading it: if, within the same true grade, test ITF is systematically lower/higher than train ITF, the
hypothesis is supported; if the distributions line up, it is not.

Usage (from the sgsq_pipeline root):
    python3 scripts/diagnose_feature_shift.py --config configs/config.yaml
    python3 scripts/diagnose_feature_shift.py --csv sgsq_data/severity/severity_labels_detector_derived_v2.csv
Writes figures/fig_feature_shift.png and prints a table you can paste back.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPLITS = ["train", "val", "test"]
COL = {"train": "#2a6f97", "val": "#8fb8d6", "test": "#b8860b"}
FEATS = [("inflamed_tooth_fraction", "ITF"), ("inflamed_area_fraction", "IAF")]


def spearman(x, y):
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--csv", default=None, help="detector-derived CSV (default: severity_dataset/severity_labels_detector_derived_v2.csv)")
    ap.add_argument("--out", default="figures/fig_feature_shift.png")
    a = ap.parse_args()
    if a.csv is None:
        import yaml
        cfg = yaml.safe_load(open(a.config))
        a.csv = str(Path(cfg["paths"]["severity_dataset"]) / "severity_labels_detector_derived_v2.csv")
    df = pd.read_csv(a.csv)
    need = {"split", "severity_label"} | {f for f, _ in FEATS}
    if need - set(df.columns):
        raise SystemExit(f"CSV is missing columns {need - set(df.columns)}; use the detector-derived CSV.")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=False)
    w = 0.26
    print(f"{'feature':5s} {'grade':5s} " + " ".join(f"{s + ' median (n)':>20s}" for s in SPLITS))
    for ax, (feat, name) in zip(axes, FEATS):
        for g in range(5):
            row = []
            for i, s in enumerate(SPLITS):
                v = df[(df["split"] == s) & (df["severity_label"] == g)][feat].values
                row.append(f"{np.median(v):.3f} ({len(v)})" if len(v) else "n/a")
                if len(v) == 0:
                    continue
                bp = ax.boxplot(v, positions=[g + (i - 1) * w], widths=w * 0.85, patch_artist=True, showfliers=False,
                                medianprops=dict(color="#1f1f1f", lw=1.0), whiskerprops=dict(color="#6b6b6b", lw=0.7),
                                capprops=dict(color="#6b6b6b", lw=0.7), boxprops=dict(lw=0.5))
                bp["boxes"][0].set_facecolor(COL[s])
            print(f"{name:5s} S{g + 1:<4d} " + " ".join(f"{r:>20s}" for r in row))
        ax.set_xticks(range(5)); ax.set_xticklabels([f"S{g + 1}" for g in range(5)], fontsize=7)
        ax.set_title(f"{name} by true severity grade", fontsize=8.5)
        ax.set_ylabel(name, fontsize=7.5); ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
        for s in SPLITS:
            d = df[df["split"] == s]
            print(f"  Spearman({name}, grade) {s}: {spearman(d[feat], d['severity_label']):.3f}")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COL[s], ec="none") for s in SPLITS]
    axes[0].legend(handles, SPLITS, fontsize=6.5, frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300)
    print("Saved", a.out)


if __name__ == "__main__":
    main()
