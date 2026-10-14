"""
scripts/make_results_figures.py
===============================
Builds the three data-only Results figures from the FINAL held-out-test numbers (README / paper tables).
No model or GPU is needed -- the values below are copied from the verified results.

  fig_results_qwk.png        forest plot: QWK with 95% bootstrap CI for every severity approach
  fig_results_pergrade.png   precision / recall / F1 per severity grade, primary model (support on top)
  fig_results_itfiaf.png     ITF/IAF ablation: validation vs test QWK (shows the ranking flip)

Usage:  python3 scripts/make_results_figures.py --outdir figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"
BLUE, BLUE_L, AMBER, GREY = "#2a6f97", "#8fb8d6", "#b8860b", "#8c8c8c"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "text.color": INK})


def clean(ax, grid_axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ QWK forest plot
ROWS = [  # label, QWK, lo, hi, kind
    ("SGSQ-Net, detector-derived ITF/IAF (primary)", 0.467, 0.329, 0.588, "primary"),
    ("EfficientNet-B0 (end-to-end)", 0.301, 0.191, 0.404, "other"),
    ("Box-derived binary ITF/IAF (GT boxes)", 0.259, 0.157, 0.355, "other"),
    ("ViT-B/16 (end-to-end)", 0.247, 0.101, 0.382, "other"),
    ("Rule-based threshold on ITF/IAF", 0.119, 0.067, 0.178, "other"),
    ("Rule-based threshold on predicted IAR", -0.102, -0.174, -0.032, "other"),
    ("Segmentation-guided (U-Net IAR + redness)", -0.111, -0.195, -0.027, "other"),
    ("GT-continuous oracle (circular, excluded)", 0.966, 0.946, 0.982, "oracle"),
]


def fig_qwk(out):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    n = len(ROWS)
    ys = np.arange(n)[::-1].astype(float)
    ys[-1] -= 0.5                                   # extra gap before the oracle row
    for (lab, q, lo, hi, kind), y in zip(ROWS, ys):
        col = BLUE if kind == "primary" else GREY
        ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="butt")
        if kind == "oracle":
            ax.plot(q, y, "o", mfc="white", mec=GREY, mew=1.4, ms=6.5)
        else:
            ax.plot(q, y, "o", color=col, ms=6.5, mec="white", mew=0.8)
        ax.text(1.04, y, f"{q:.3f} [{lo:.3f}, {hi:.3f}]".replace("-", "−"), va="center",
                ha="left", fontsize=7.5, color=INK, transform=ax.get_yaxis_transform(),
                weight="bold" if kind == "primary" else "normal")
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.text(0.003, ys[0] + 0.62, "chance-level agreement", fontsize=6.5, color=MUTED, va="bottom")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in ROWS], fontsize=7.5)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlim(-0.3, 1.02)
    ax.set_ylim(ys[-1] - 0.7, ys[0] + 1.0)
    ax.set_xlabel("Quadratic weighted kappa on the test split (95% bootstrap CI)")
    clean(ax, "x")
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.36, right=0.80, top=0.97, bottom=0.17)
    fig.savefig(out, dpi=300); plt.close(fig); print("Saved", out)


# ------------------------------------------------------------------ per-grade bars
def fig_pergrade(out):
    grades = ["S1", "S2", "S3", "S4", "S5"]
    P = [0.27, 0.43, 0.37, 0.25, 0.30]
    R = [0.76, 0.07, 0.27, 0.21, 0.67]
    F = [0.39, 0.13, 0.31, 0.23, 0.41]
    N = [34, 80, 37, 19, 12]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x, w = np.arange(5), 0.26
    for i, (v, lab, c) in enumerate([(P, "precision", BLUE_L), (R, "recall", BLUE), (F, "F1", AMBER)]):
        ax.bar(x + (i - 1) * w, v, w, color=c, label=lab, lw=0)
    for xi, n in zip(x, N):
        ax.text(xi, 0.90, f"n={n}", ha="center", fontsize=6.5, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(grades)
    ax.set_ylim(0, 1.0); ax.set_ylabel("score")
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              handlelength=1.0, columnspacing=1.0)
    clean(ax, "y")
    fig.tight_layout()
    fig.savefig(out, dpi=300); plt.close(fig); print("Saved", out)


# ------------------------------------------------------------------ ITF/IAF ablation
def fig_itfiaf(out):
    rows = [  # label, val, test, lo, hi
        ("ITF only", 0.552, 0.532, 0.402, 0.634),
        ("IAF only", 0.558, 0.501, 0.357, 0.614),
        ("ITF + IAF (retrained)", 0.530, 0.419, 0.270, 0.545),
        ("ITF + IAF (primary checkpoint)", None, 0.467, 0.329, 0.588),
    ]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ys = np.arange(len(rows))[::-1]
    for (lab, v, t, lo, hi), y in zip(rows, ys):
        ax.plot([lo, hi], [y, y], color=BLUE, lw=1.6, solid_capstyle="butt")
        ax.plot(t, y, "o", color=BLUE, ms=6, mec="white", mew=0.8)
        if v is not None:
            ax.plot(v, y, "D", mfc="white", mec=AMBER, mew=1.4, ms=5.5)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlim(0.2, 0.7)
    ax.set_xlabel("QWK")
    ax.plot([], [], "D", mfc="white", mec=AMBER, mew=1.4, ms=5.5, ls="none", label="validation")
    ax.plot([], [], "o", color=BLUE, ms=6, ls="none", label="test (95% CI)")
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.35, 1.17),
              handletextpad=0.3, columnspacing=1.0)
    clean(ax, "x"); ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=300); plt.close(fig); print("Saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    o = Path(a.outdir); o.mkdir(parents=True, exist_ok=True)
    fig_qwk(o / "fig_results_qwk.png")
    fig_pergrade(o / "fig_results_pergrade.png")
    fig_itfiaf(o / "fig_results_itfiaf.png")
