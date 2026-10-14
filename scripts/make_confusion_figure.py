"""
scripts/make_confusion_figure.py
================================
Confusion-matrix figure from the REAL held-out-test run (no retraining, no GPU).
Reads checkpoints/test_set_full_report.json, written by scripts/evaluate_test_set.py
(key "confusion_matrices": {method name: 5x5 counts, rows = true grade, cols = predicted grade}).

Draws the primary (detector-derived) model next to the segmentation-guided pathway, row-normalised
(each row sums to 1) with raw counts annotated, plus an ordinal-error summary printed to the console
(exact / off-by-one / off-by-two-or-more), which is useful for the text.

Usage (from the sgsq_pipeline root):
    python3 scripts/make_confusion_figure.py --report checkpoints/test_set_full_report.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

PRIMARY_KEY = "DETECTOR-DERIVED"          # substring that identifies the primary method
SEG_KEY = "segmentation-guided"           # substring that identifies the segmentation-guided method
CMAP = LinearSegmentedColormap.from_list("blue1", ["#f4f8fb", "#2a6f97"])


def find(cms, needle):
    for k in cms:
        if needle.lower() in k.lower():
            return k
    return None


def ordinal_summary(cm):
    cm = np.asarray(cm)
    n = cm.sum()
    idx = np.arange(cm.shape[0])
    d = np.abs(idx[:, None] - idx[None, :])
    return {"exact": cm[d == 0].sum() / n, "off_by_1": cm[d == 1].sum() / n, "off_by_2plus": cm[d >= 2].sum() / n}


def draw(ax, cm, title):
    cm = np.asarray(cm, float)
    rows = cm.sum(1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    ax.imshow(norm, cmap=CMAP, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.55 else "#1f1f1f")
    labs = [f"S{k + 1}" for k in range(cm.shape[0])]
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=7)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlabel("predicted grade", fontsize=7.5); ax.set_ylabel("true grade", fontsize=7.5)
    ax.set_title(title, fontsize=8)
    ax.set_xticks(np.arange(-.5, len(labs)), minor=True); ax.set_yticks(np.arange(-.5, len(labs)), minor=True)
    ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="both", length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="checkpoints/test_set_full_report.json")
    ap.add_argument("--out", default="figures/fig_confusion_matrix.png")
    a = ap.parse_args()
    rep = json.load(open(a.report))
    # evaluate_test_set.py nests it as report["severity"]["confusion_matrices"]; accept a top-level key too
    cms = (rep.get("severity") or {}).get("confusion_matrices") or rep.get("confusion_matrices")
    if not cms:
        raise SystemExit("No 'confusion_matrices' under report['severity'] -- your evaluate_test_set.py is the older version "
                         "without the confusion-matrix block; replace it with the latest and re-run it.")
    kp, ks = find(cms, PRIMARY_KEY), find(cms, SEG_KEY)
    if kp is None:
        raise SystemExit(f"Primary method not found. Available: {list(cms)}")
    panels = [(kp, "(a) Primary: detector-derived ITF/IAF")]
    if ks:
        panels.append((ks, "(b) Segmentation-guided pathway"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.1))
    axes = np.atleast_1d(axes)
    for ax, (k, t) in zip(axes, panels):
        draw(ax, cms[k], t)
        s = ordinal_summary(cms[k])
        print(f"{k}\n  exact {s['exact']:.3f} | off-by-one {s['off_by_1']:.3f} | off-by-two-or-more {s['off_by_2plus']:.3f}")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300)
    print("Saved", a.out)


if __name__ == "__main__":
    main()
