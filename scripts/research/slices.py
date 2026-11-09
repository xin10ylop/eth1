"""When is ETH most predictable?  IC of cached walk-forward predictions by regime slice.

    python scripts/research/slices.py s1 lean_cb_xp_s5 2025-11-2026-01-2026-03 [f1 ...]
Slices: volatility tercile (sigma vs its trailing 1-day median), weekend/weekday, UTC session.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from ethpredict.config import FEAT, PREDS  # noqa: E402

args = sys.argv[1:]
panel = pd.read_parquet(FEAT / "panel.parquet", columns=["ts", "sigma", "y_s1", "y_f1", "y_s5", "y_f5"])
med = panel["sigma"].rolling(1440, min_periods=200).median()
panel["vrel"] = np.log(panel["sigma"] / med)
rows = {}
for i in range(0, len(args), 3):
    key, tag, months = args[i:i + 3]
    p = pd.read_parquet(PREDS / f"lgbm_{key}_{tag}_{months}.parquet")[["ts", "pred"]]
    d = p.merge(panel, on="ts").dropna(subset=[f"y_{key}", "pred", "vrel"])
    T = d["ts"] + pd.Timedelta(minutes=1)
    q = d["vrel"].quantile([1 / 3, 2 / 3]).values
    sl = {
        "all": np.ones(len(d), bool),
        "low-vol tercile": d["vrel"] <= q[0], "mid-vol tercile": (d["vrel"] > q[0]) & (d["vrel"] <= q[1]),
        "high-vol tercile": d["vrel"] > q[1],
        "weekend": T.dt.dayofweek >= 5, "weekday": T.dt.dayofweek < 5,
        "Asia 00-08 UTC": T.dt.hour < 8, "Europe 08-13 UTC": (T.dt.hour >= 8) & (T.dt.hour < 13),
        "US 13-21 UTC": (T.dt.hour >= 13) & (T.dt.hour < 21), "late US 21-24 UTC": T.dt.hour >= 21,
    }
    rows[f"{key} {tag}"] = {k: np.corrcoef(d.loc[m, "pred"], d.loc[m, f"y_{key}"])[0, 1] for k, m in sl.items()}
print(pd.DataFrame(rows).round(3).to_string())
