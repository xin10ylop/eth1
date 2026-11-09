"""Measure live-vs-final feature skew: features computed at T+1s (production timing) vs the same
minute recomputed 3 minutes later from fresh, complete API data."""
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.simplefilter("ignore")
from ethpredict.live import sources as live  # noqa: E402
from ethpredict.live.predictor import Predictor  # noqa: E402
from ethpredict.live.cli import clock_offset_s  # noqa: E402

n_min = int(sys.argv[1]) if len(sys.argv) > 1 else 5
variant = sys.argv[2] if len(sys.argv) > 2 else "spot"
delay = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
pred = Predictor(variant)
off = clock_offset_s()
src = live.CachedSource(live.RestSource(use_futures=pred.uses_futures))
rows = []
for k in range(n_min):
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) + pd.Timedelta(seconds=off)
    T = now.floor("min") + pd.Timedelta(minutes=1)
    time.sleep(max(0, (T + pd.Timedelta(seconds=delay) - now).total_seconds()))
    dfh, A = live.build_features(src, T, use_futures=pred.uses_futures, last_n=1, wide=pred.needs_wide)
    X = live.model_inputs(dfh, A)
    rows.append((T, X.iloc[-1], pred.raw(X)))
    print("live", T, flush=True)
time.sleep(180)
diffs, zd = {}, []
for T, xl, rl in rows:
    fresh = live.RestSource(use_futures=pred.uses_futures)
    dfh, A = live.build_features(fresh, T, use_futures=pred.uses_futures, last_n=1, wide=pred.needs_wide)
    X = live.model_inputs(dfh, A)
    xf = X.iloc[-1]
    rf = pred.raw(X)
    cols = [c for c in xl.index if c in xf.index]
    err = ((xl[cols].astype(float) - xf[cols].astype(float)).abs() / (xf[cols].astype(float).abs() + 1e-3)).fillna(0)
    for c, e in err[err > 1e-3].items():
        diffs.setdefault(c, []).append(e)
    for key in rl:
        zd.append((T, key, rl[key]["sat"][0], rf[key]["sat"][0]))
print("features deviating at forecast time vs final data (count of minutes, max rel err):")
for c in sorted(diffs, key=lambda c: -len(diffs[c]))[:25]:
    print(f"   {c:28s} n={len(diffs[c])} max={max(diffs[c]):.3f}")
z = pd.DataFrame(zd, columns=["T", "key", "z_live", "z_final"])
print(z.assign(dz=lambda d: d.z_live - d.z_final).round(4).to_string())
