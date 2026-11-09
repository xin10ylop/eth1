"""Build all model-matrix files from the panel (run after `python -m ethpredict.features.panel`).

Writes data/features/{X, Xalt, Xalt2, Xcb}.parquet, row-aligned with the panel (ts >= START).
    python scripts/pipeline/build_features.py START END [parts]     parts: comma list of X,Xalt,Xalt2,Xcb (default all)
"""
import sys
import time
import warnings

import pandas as pd

sys.path.insert(0, ".")
warnings.simplefilter("ignore")
from ethpredict.features import coinbase  # noqa: E402
from ethpredict.features import crossasset as altfeat  # noqa: E402
from ethpredict.features import matrix as F  # noqa: E402
from ethpredict.config import COINBASE_FILE, FEAT, SEC  # noqa: E402

START = "2025-05-03"   # first two days are warm-up for 24h rolling statistics
start, end = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("2025-05-01", "2026-11-02")
parts = sys.argv[3].split(",") if len(sys.argv) > 3 else ["X", "Xalt", "Xalt2", "Xcb"]
t = time.time()
panel = pd.read_parquet(FEAT / "panel.parquet")
panel = panel[panel["ts"] >= START].reset_index(drop=True)
if "X" in parts:
    F.model_matrix(panel).to_parquet(FEAT / "X.parquet")
    print("X", round(time.time() - t), "s", flush=True)
for name, fn in (("Xalt", altfeat.build), ("Xalt2", altfeat.build2)):
    if name not in parts:
        continue
    d = fn(start, end)
    d = panel[["ts"]].merge(d, on="ts", how="left")
    d.drop(columns="ts").to_parquet(FEAT / f"{name}.parquet")
    print(name, d.shape, round(time.time() - t), "s", flush=True)
if "Xcb" in parts:
    cb = pd.read_parquet(SEC / "coinbase" / COINBASE_FILE)
    coinbase.features(panel[["ts", "s_close", "sigma", "s_vol60"]], cb).to_parquet(FEAT / "Xcb.parquet")
    print("Xcb done", round(time.time() - t), "s")
