"""
scripts/make_dataset_figures.py
===============================
Builds the paper's DATASET figures/tables from the REAL converted dataset (nothing hand-drawn).

  fig_dataset_samples.png      3 x 5 grid: one random TRAIN image per severity grade S1-S5
                               (row 1 photograph, row 2 ground-truth tooth boxes coloured by MGI,
                                row 3 the coarse box-derived mask used to train the segmenters)
  fig_dataset_distribution.png (a) images per severity grade and split, (b) teeth per MGI grade and split
  dataset_stats.csv / dataset_stats_table.tex   numbers + a ready-to-paste LaTeX table

Image choice (no cherry-picking): a random image per grade with a fixed --seed (printed, so the
caption can state the rule).

Usage (from the sgsq_pipeline root):
    python3 scripts/make_dataset_figures.py --config configs/config.yaml
    python3 scripts/make_dataset_figures.py --config configs/config.yaml --seed 3 --split train
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Rectangle, Patch
from PIL import Image

RAMP = ["#ffe9b0", "#fdc16a", "#f48c3f", "#d9532b", "#9c1d1d"]   # MGI 0..4 (same as Fig. 1)
GINGIVA_COL = "#1b8a8f"
TOOTH = {2, 3, 4, 5, 6}                                           # class id - 2 = MGI
SPLITS = ["train", "val", "test"]
CELL_ASPECT = 1.6                                                 # width / height of each sample


def read_labels(path):
    rows = []
    if Path(path).exists():
        for ln in Path(path).read_text().splitlines():
            p = ln.split()
            if len(p) >= 5:
                rows.append((int(p[0]), *[float(v) for v in p[1:5]]))
    return rows                                                   # (cls, cx, cy, w, h) normalized


def crop_window(boxes, W, H, mx=0.08, my=0.25):
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    dx, dy = (x1 - x0) * mx, (y1 - y0) * my
    x0, x1, y0, y1 = max(0, x0 - dx), min(W, x1 + dx), max(0, y0 - dy), min(H, y1 + dy)
    w, h = x1 - x0, y1 - y0
    if w / h < CELL_ASPECT:                       # too tall -> widen
        nw = min(W, h * CELL_ASPECT); c = (x0 + x1) / 2
        x0, x1 = max(0, c - nw / 2), min(W, c + nw / 2)
    else:                                         # too wide -> heighten
        nh = min(H, w / CELL_ASPECT); c = (y0 + y1) / 2
        y0, y1 = max(0, c - nh / 2), min(H, c + nh / 2)
    return x0, x1, y0, y1


SPLIT_DIRS = {"train": "Training", "val": "Validation", "test": "Test"}
LABEL_ROOT = None   # set in main(); folder holding the 7-class (MGI) labels


def find_label(stem, split):
    """Locate the 7-class label file (class ids 0-1 jaw, 2-6 tooth MGI 0-4) for one image."""
    r = LABEL_ROOT
    for cand in (r / "labels" / split / f"{stem}.txt",                 # YOLO-style layout
                 r / split / f"{stem}.txt",
                 r / SPLIT_DIRS[split] / "Labels" / f"{stem}.txt",     # original Mendeley layout
                 r / SPLIT_DIRS[split] / "labels" / f"{stem}.txt"):
        if cand.exists():
            return cand
    return None


def stem_of(filename, split):
    s = Path(filename).stem
    return s[len(split) + 1:] if s.startswith(split + "_") else s


def collect_stats(df):
    """Per-split image / tooth counts from the real label files."""
    tooth_counts = {s: np.zeros(5, int) for s in SPLITS}
    missing = []
    for _, r in df.iterrows():
        lp = find_label(stem_of(r["filename"], r["split"]), r["split"])
        if lp is None:
            missing.append(r["filename"]); continue
        for c, *_ in read_labels(lp):
            if c in TOOTH:
                tooth_counts[r["split"]][c - 2] += 1
    if missing:
        raise SystemExit(f"{len(missing)} of {len(df)} images have no label file under {LABEL_ROOT} "
                         f"(e.g. {missing[0]}). Pass --labels-root pointing to the folder with the 7-class labels "
                         f"(the original Mendeley 'Dataset' folder works).")
    if sum(int(v.sum()) for v in tooth_counts.values()) == 0:
        raise SystemExit("Label files were found but contain no tooth classes 2-6. They look like the 2-class "
                         "(tooth/gingiva) conversion; pass --labels-root pointing to the ORIGINAL Mendeley 'Dataset' "
                         "folder or to the 7-class detection labels used to train the MGI detector.")
    out = {}
    for s in SPLITS:
        d = df[df["split"] == s]
        out[s] = dict(images=len(d),
                      teeth=int(tooth_counts[s].sum()),
                      grades=[int((d["severity_label"] == g).sum()) for g in range(5)],
                      mgi=tooth_counts[s].tolist(),
                      mean_mgi=float(d["mean_mgi"].mean()),
                      teeth_per_img=float(tooth_counts[s].sum() / max(len(d), 1)))
    return out


def fig_samples(df, split, seed, seg_root, out):
    rng = np.random.default_rng(seed)
    d = df[df["split"] == split]
    fig, axes = plt.subplots(3, 5, figsize=(7.4, 4.3))
    for g in range(5):
        pool = d[d["severity_label"] == g].reset_index(drop=True)
        r = pool.iloc[int(rng.integers(len(pool)))]
        stem = stem_of(r["filename"], split)
        print(f"S{g + 1}: {r['filename']}  mean MGI = {r['mean_mgi']:.2f}")
        img = Image.open(seg_root / "images" / r["filename"]).convert("RGB")
        W, H = img.size
        labs = read_labels(find_label(stem, split))
        xyxy = [((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
                for _, cx, cy, w, h in labs]
        x0, x1, y0, y1 = crop_window(xyxy, W, H)
        mask_p = seg_root / "masks" / (Path(r["filename"]).stem + ".png")
        mask = np.array(Image.open(mask_p).convert("L")) > 127 if mask_p.exists() else np.zeros((H, W), bool)
        for row in range(3):
            ax = axes[row, g]
            ax.imshow(img); ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
            if row == 2:
                ov = np.zeros((H, W, 4)); ov[mask] = (0.85, 0.15, 0.15, 0.55)
                ax.imshow(ov)
            if row == 1:
                for (c, *_), (a, b, c2, d2) in zip(labs, xyxy):
                    tooth = c in TOOTH
                    col = RAMP[c - 2] if tooth else GINGIVA_COL
                    ax.add_patch(Rectangle((a, b), c2 - a, d2 - b, fill=False, ec=col,
                                           lw=1.1 if tooth else 0.9, ls="-" if tooth else "--"))
                    if tooth:
                        ax.text(a + 2, b + 2, str(c - 2), fontsize=5, va="top", ha="left", weight="bold",
                                bbox=dict(boxstyle="square,pad=0.1", fc=col, ec="none"))
            if row == 0:
                ax.set_title(f"S{g + 1} (mean MGI {r['mean_mgi']:.2f})", fontsize=7.5, pad=3)
    handles = [Patch(fc=RAMP[k], ec="none", label=f"MGI {k}") for k in range(5)]
    handles.append(Patch(fc="none", ec=GINGIVA_COL, ls="--", label="Jaw region"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=6.5,
               handlelength=1.1, columnspacing=1.0, handletextpad=0.3)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.94, bottom=0.07, wspace=0.03, hspace=0.04)
    fig.canvas.draw()
    for row, lab in enumerate(["Photograph", "Annotated teeth", "Box-derived mask"]):
        p_ = axes[row, 0].get_position()
        fig.text(0.012, (p_.y0 + p_.y1) / 2, lab, rotation=90, va="center", ha="center", fontsize=7.5)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Saved", out)


def fig_distribution(stats, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6))
    cols = {"train": "#2a6f97", "val": "#8fb8d6", "test": "#b8860b"}
    w = 0.26
    for ax, key, labels, title in [
        (axes[0], "grades", [f"S{g + 1}" for g in range(5)], "(a) Images per severity grade"),
        (axes[1], "mgi", [f"MGI {k}" for k in range(5)], "(b) Teeth per MGI grade")]:
        x = np.arange(5)
        for i, s in enumerate(SPLITS):
            v = np.array(stats[s][key], float)
            ax.bar(x + (i - 1) * w, v, w, color=cols[s], label=f"{s} (n={stats[s]['images']})")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(title, fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("count", fontsize=7.5)
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.28)
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper left", ncol=3, columnspacing=1.0, handlelength=1.2)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Saved", out)


def write_table(stats, out_csv, out_tex):
    rows = []
    for s in SPLITS:
        st = stats[s]
        rows.append([s, st["images"], st["teeth"], f"{st['teeth_per_img']:.1f}", *st["grades"], *st["mgi"]])
    cols = ["split", "images", "teeth", "teeth/img"] + [f"S{g + 1}" for g in range(5)] + [f"MGI{k}" for k in range(5)]
    pd.DataFrame(rows, columns=cols).to_csv(out_csv, index=False)
    tot = [sum(stats[s]["images"] for s in SPLITS), sum(stats[s]["teeth"] for s in SPLITS)]
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Dataset composition by split: images per severity grade (S1--S5) and annotated teeth per MGI grade. "
           r"Source: Mendeley dataset~\cite{duy2024dental}; grades follow Section~\ref{sec:label-construction}, MGI follows Lobene et al.~\cite{lobene1986modified}.}",
           r"\label{tab:dataset-composition}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
           r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}lrrrrrrrrrrrr@{}}", r"\toprule",
           r" & & & \multicolumn{5}{c}{Images per grade} & \multicolumn{5}{c}{Teeth per MGI} \\",
           r"\cmidrule(lr){4-8}\cmidrule(lr){9-13}",
           r"Split & Images & Teeth & S1 & S2 & S3 & S4 & S5 & 0 & 1 & 2 & 3 & 4 \\", r"\midrule"]
    for s in SPLITS:
        st = stats[s]
        tex.append(f"{s.capitalize()} & {st['images']} & {st['teeth']} & " +
                   " & ".join(str(v) for v in st["grades"]) + " & " + " & ".join(str(v) for v in st["mgi"]) + r" \\")
    tex.append(r"\midrule")
    tg = [sum(stats[s]["grades"][g] for s in SPLITS) for g in range(5)]
    tm = [sum(stats[s]["mgi"][k] for s in SPLITS) for k in range(5)]
    tex.append(f"Total & {tot[0]} & {tot[1]} & " + " & ".join(map(str, tg)) + " & " + " & ".join(map(str, tm)) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    Path(out_tex).write_text("\n".join(tex) + "\n")
    print("Saved", out_csv, "and", out_tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--csv", default="severity_labels.csv")
    ap.add_argument("--split", default="train", choices=SPLITS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels-root", default=None,
                    help="folder with the 7-class labels (class 0-1 jaw, 2-6 tooth MGI 0-4): the original Mendeley "
                         "'Dataset' folder or a YOLO-style folder with labels/<split>/. Default: detection_dataset from config.")
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    global LABEL_ROOT
    det_root = Path(cfg["paths"]["detection_dataset"])
    LABEL_ROOT = Path(a.labels_root) if a.labels_root else det_root
    seg_root = Path(cfg["paths"]["segmentation_dataset"])
    df = pd.read_csv(Path(cfg["paths"]["severity_dataset"]) / a.csv)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    stats = collect_stats(df)
    for s in SPLITS:
        print(s, stats[s])
    edges = np.quantile(df.loc[df["split"] == "train", "mean_mgi"], [0.2, 0.4, 0.6, 0.8])
    print("Quantile edges of mean MGI (train):", np.round(edges, 3).tolist())
    print("Selection rule for caption: one random", a.split, "image per grade, seed =", a.seed)

    write_table(stats, out / "dataset_stats.csv", out / "dataset_stats_table.tex")
    fig_distribution(stats, out / "fig_dataset_distribution.png")
    fig_samples(df, a.split, a.seed, seg_root, out / "fig_dataset_samples.png")


if __name__ == "__main__":
    main()
