"""Training/serving skew check: compare logged live feature vectors with the same minutes rebuilt
from the official historical files (run after the day's files are published on data.binance.vision
and downloaded with ethpredict.data.binance; Coinbase candles are re-fetched).

python scripts/live/check_live_skew.py logs/live_features_spot.csv
"""
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.simplefilter("ignore")
from ethpredict.live import sources as live  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else str(LOGS / "live_features_spot.csv")
use_futures = "full" in path
L = pd.read_csv(path)
src = live.ReplaySource()
worst = {}
for _, row in L.iterrows():
    T = pd.Timestamp(row["T"])
    dfh, A = live.build_features(src, T, use_futures=use_futures, last_n=1)
    B = live.model_inputs(dfh, A).iloc[-1]
    cols = [c for c in L.columns if c != "T" and c in B.index]
    a, b = row[cols].astype(float), B[cols].astype(float)
    err = ((a - b).abs() / (b.abs() + 1e-3)).fillna(0)
    for c, e in err[err > 1e-2].items():
        worst[c] = max(worst.get(c, 0), e)
print(f"{len(L)} live minutes compared; features with >1% relative deviation (worst first):")
for c in sorted(worst, key=worst.get, reverse=True)[:25]:
    print(f"   {c:28s} {worst[c]:.3f}")
