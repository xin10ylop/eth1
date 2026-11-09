"""Paired day-block bootstrap of IC differences between cached walk-forward predictions.

    python scripts/research/compare_preds.py s1 2025-11-2026-01-2026-03 sat_all base core loo_cb ...
Compares every listed tag against the first one (reference). Positive diff = reference is better.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from ethpredict.config import FEAT, PREDS  # noqa: E402
from ethpredict.modeling import evaluate as E  # noqa: E402

key, months, ref, *others = sys.argv[1:]
venue, h = key[0], key[1:]
if h.endswith("s"):      # Ysub is row-aligned with the panel from 2025-05-03
    panel = pd.read_parquet(FEAT / "panel.parquet", columns=["ts"])
    panel = panel[panel["ts"] >= "2025-05-03"].reset_index(drop=True)
    panel[f"y_{key}"] = pd.read_parquet(FEAT / "Ysub.parquet")[f"y_{key}"].values
else:
    panel = pd.read_parquet(FEAT / "panel.parquet", columns=["ts", f"y_{key}"])
ycol = f"y_{key}"


def load(tag):
    return pd.read_parquet(PREDS / f"lgbm_{key}_{tag}_{months}.parquet")[["ts", "pred"]].rename(columns={"pred": tag})


d = load(ref)
for t in others:
    d = d.merge(load(t), on="ts")
d = d.merge(panel[["ts", ycol]], on="ts").dropna()
print(f"{key} {months}: n={len(d)}  reference={ref}  IC(ref)={np.corrcoef(d[ref], d[ycol])[0, 1]:.4f}")
for t in others:
    r = E.paired_block_bootstrap(d["ts"], d[ycol].values, d[t].values, d[ref].values)
    mi = d.assign(m=d.ts.dt.strftime("%Y-%m")).groupby("m").apply(
        lambda g: np.corrcoef(g[ref], g[ycol])[0, 1] - np.corrcoef(g[t], g[ycol])[0, 1], include_groups=False)
    print(f"  {ref} - {t:14s} dIC {r['diff']:+.4f}  CI95 [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  P(ref not better) {r['p_le_0']:.3f}"
          f"   months ref>other: {int((mi > 0).sum())}/{len(mi)}")
