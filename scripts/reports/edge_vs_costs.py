"""Gross edge per trade (bp) by confidence level vs realistic costs, on holdout predictions.

A 'trade' = take the forecast direction at the forecast instant (+delay seconds) and close after h
minutes. Gross edge = mean(sign(forecast) * realized log return). Round-trip cost references:
  Binance USD-M perp taker 0.05%/side (VIP0) -> 10 bp round trip; maker 0.02%/side -> 4 bp;
  market makers on rebate tiers ~ 0 bp or better.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from ethpredict.config import FEAT, HOLDOUT, PREDS  # noqa: E402

keys = (sys.argv[1] if len(sys.argv) > 1 else "s1,f1,s5,f5").split(",")
pattern = sys.argv[2] if len(sys.argv) > 2 else "final_full_{key}_holdout"
cols = ["ts"] + [f"y_{k}" for k in keys] + [f"y_{k}_d{d}" for k in keys for d in (2, 5)]
pcols = [c for c in cols if not any(c.startswith(f"y_{k}") for k in keys if k.endswith("s"))]
df = pd.read_parquet(FEAT / "panel.parquet", columns=pcols)
df = df[df["ts"] >= "2025-05-03"].reset_index(drop=True)
if any(k.endswith("s") for k in keys):
    df = pd.concat([df, pd.read_parquet(FEAT / "Ysub.parquet")], axis=1)
df = df.set_index("ts")
out = {}
for k in keys:
    p = pd.read_parquet(PREDS / (pattern.format(key=k) + ".parquet")).set_index("ts")["pred"]
    d = df.loc[p.index]
    a = p.abs()
    res = {}
    for q in (0.0, 0.9, 0.99, 0.999):
        sel = a >= a.quantile(q)
        r = {}
        for dl in (0, 2, 5):
            y = d[f"y_{k}" if dl == 0 else f"y_{k}_d{dl}"][sel]
            g = (np.sign(p[sel]) * y * 1e4).dropna()
            r[f"edge_bp_d{dl}"] = round(float(g.mean()), 3)
        r["trades_per_day"] = round(float(sel.sum() / (len(p) / 1440)), 1)
        r["avg_abs_move_bp"] = round(float((d[f"y_{k}"][sel].abs() * 1e4).mean()), 2)
        res[f"top{100 * (1 - q):g}%"] = r
    out[k] = res
print(json.dumps(out, indent=1))
(HOLDOUT / "edge_vs_costs.json").write_text(json.dumps(out, indent=1))
