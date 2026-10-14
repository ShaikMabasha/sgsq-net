"""
scripts/check_pipeline_facts.py
===============================
Answers three factual questions the paper needs, using the REAL final detector (no retraining):

  Q1  How often is the BINARY feature (share of predicted teeth with MGI >= 1) exactly 1.0?
      -> supports/refutes "the binary feature was (almost) always 1.0".
  Q2  Do the features stored in the severity CSV (used to train the head) match features recomputed now with the
      final detector on the same images, in every split (incl. TRAIN)?
      -> confirms the head was trained on features from the same final detector.
  Q3  How many images have NO tooth detections (n = 0) at confidence 0.25, per split, and are they in the CSV?

Usage (from the sgsq_pipeline root):
    python3 scripts/check_pipeline_facts.py --config configs/config.yaml
Writes figures/pipeline_facts_per_image.csv and prints a summary to paste back.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    import yaml
    from ultralytics import YOLO
    sys.path.append(str(Path(__file__).resolve().parent))
    from make_fig1_real_example import itf_iaf, TOOTH_CLASSES, MGI_OFFSET, CONF  # same helpers as the paper figures

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--csv", default="severity_labels_detector_derived_v2.csv")
    ap.add_argument("--detector-ckpt", default="detection_mgi_best.pt")
    ap.add_argument("--out", default="figures/pipeline_facts_per_image.csv")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    ckpt = Path(cfg["paths"]["checkpoints"])
    img_dir = Path(cfg["paths"]["segmentation_dataset"]) / "images"
    df = pd.read_csv(Path(cfg["paths"]["severity_dataset"]) / a.csv)
    det = YOLO(str(ckpt / a.detector_ckpt))
    print(f"CSV rows: {len(df)}; splits: {df['split'].value_counts().to_dict()}; confidence = {CONF}")

    rows = []
    for _, r in df.iterrows():
        res = det.predict(str(img_dir / r["filename"]), verbose=False, conf=CONF)[0]
        cls = res.boxes.cls.cpu().numpy().astype(int)
        xywhn = res.boxes.xywhn.cpu().numpy()
        t = np.isin(cls, list(TOOTH_CLASSES))
        n = int(t.sum())
        if n == 0:
            rows.append(dict(filename=r["filename"], split=r["split"], n_teeth=0, binary_share=np.nan,
                             itf_now=np.nan, iaf_now=np.nan, itf_csv=r["inflamed_tooth_fraction"],
                             iaf_csv=r["inflamed_area_fraction"]))
            continue
        grades = cls[t] - MGI_OFFSET
        itf, iaf = itf_iaf(cls[t], xywhn[t])
        rows.append(dict(filename=r["filename"], split=r["split"], n_teeth=n,
                         binary_share=float((grades >= 1).mean()), itf_now=itf, iaf_now=iaf,
                         itf_csv=r["inflamed_tooth_fraction"], iaf_csv=r["inflamed_area_fraction"]))
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)

    print("\n=== Q1: binary feature (share of predicted teeth with MGI >= 1) ===")
    for s in ["train", "val", "test"]:
        d = out[(out["split"] == s) & (out["n_teeth"] > 0)]
        print(f"{s:5s} images with detections {len(d):4d} | binary == 1.0: {int((d['binary_share'] == 1.0).sum()):4d} "
              f"({100 * (d['binary_share'] == 1.0).mean():.1f}%) | min {d['binary_share'].min():.3f} | median {d['binary_share'].median():.3f}")

    print("\n=== Q2: CSV features vs. recomputed with the final detector ===")
    for s in ["train", "val", "test"]:
        d = out[(out["split"] == s) & (out["n_teeth"] > 0)]
        di, da = (d["itf_now"] - d["itf_csv"]).abs(), (d["iaf_now"] - d["iaf_csv"]).abs()
        print(f"{s:5s} max|dITF| {di.max():.5f} mean {di.mean():.5f} | max|dIAF| {da.max():.5f} mean {da.mean():.5f} "
              f"| images differing > 1e-3: ITF {int((di > 1e-3).sum())}, IAF {int((da > 1e-3).sum())} of {len(d)}")

    print("\n=== Q3: images with NO tooth detections (n = 0) ===")
    for s in ["train", "val", "test"]:
        d = out[out["split"] == s]
        print(f"{s:5s} total {len(d):4d} | n = 0: {int((d['n_teeth'] == 0).sum())} | CSV feature values for those: "
              f"{d[d['n_teeth'] == 0][['itf_csv', 'iaf_csv']].round(3).values.tolist()[:5]}")
    print("\nSaved", a.out)


if __name__ == "__main__":
    main()
