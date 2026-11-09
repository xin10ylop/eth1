"""Verify that the live feature builder reproduces the training features exactly.

For random historical instants T, rebuild the feature row from the raw inputs a live process would
have had (minute klines for history, per-second bars for the recent window) via
ethpredict.live.sources.build_features(ReplaySource) and compare with the training matrix.
Also checks that perp 1m klines (ethum1m files) equal our aggTrade-derived minute aggregates.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.simplefilter("ignore")
from ethpredict.features import matrix as F  # noqa: E402
from ethpredict.live import sources as live  # noqa: E402
from ethpredict.config import DIR_PERP, FEAT, MODELS, SEC  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
panel = pd.read_parquet(FEAT / "panel.parquet")
panel = panel[panel["ts"] >= "2025-05-03"].reset_index(drop=True)
X = pd.concat([pd.read_parquet(FEAT / "X.parquet"), F.extra_features(panel), pd.read_parquet(FEAT / "Xalt.parquet")], axis=1)
rng = np.random.default_rng(1)
idx = rng.choice(np.arange(5000, len(panel) - 10), n, replace=False)
src = live.ReplaySource()
worst = {}
for i in idx:
    T = panel["ts"].iloc[i] + pd.Timedelta(minutes=1)
    t = time.time()
    dfh, A = live.build_features(src, T, use_futures=True, last_n=1)
    Xl = live.model_inputs(dfh, A).iloc[-1]
    Xt = X.iloc[i]
    cols = [c for c in Xt.index if c in Xl.index]
    a, b = Xl[cols].astype(float), Xt[cols].astype(float)
    both_nan = a.isna() & b.isna()
    err = ((a - b).abs() / (b.abs() + 1e-3)).where(~both_nan, 0).fillna(1.0)
    bad = err[err > 2e-3].sort_values(ascending=False)
    print(f"T={T}  {len(cols)} cols  max rel err {err.max():.2e}  n_bad={len(bad)}  {time.time()-t:.1f}s", flush=True)
    for c, e in bad.head(8).items():
        print(f"     {c:24s} live={a[c]:.6g} train={b[c]:.6g}")
        worst[c] = max(worst.get(c, 0), e)
    missing = [c for c in Xt.index if c not in Xl.index]
    if missing:
        print("   missing in live:", missing[:20])
print("columns with any mismatch:", sorted(worst, key=worst.get, reverse=True)[:30])

# klines vs aggTrade aggregation (perp): live history uses klines, training uses aggTrades
day = "2026-03-10"
if not (SEC / "ethum1m" / f"{day}.parquet").exists():
    import datetime as _dt
    from ethpredict.data import binance as _B
    (SEC / "ethum1m").mkdir(exist_ok=True)
    _B.JOBS["ethum1m"](_dt.date.fromisoformat(day)).to_parquet(SEC / "ethum1m" / f"{day}.parquet")
k = pd.read_parquet(SEC / "ethum1m" / f"{day}.parquet")
s = pd.read_parquet(SEC / DIR_PERP / f"{day}.parquet")
g = np.arange(86400) // 60
agg = pd.DataFrame({"v": s["v"].groupby(g).sum(), "bv": s["bv"].groupby(g).sum(), "c": s["c"].groupby(g).last()})
print("klines vs aggTrades  max|dv|", float(np.max(np.abs(k["volume"].values - agg["v"].values))),
      " max|dbv|", float(np.max(np.abs(k["taker_buy_volume"].values - agg["bv"].values))),
      " max|dclose|", float(np.nanmax(np.abs(k["close"].values - agg["c"].values))))

# end-to-end: the live forecaster's model scores must equal the boosters applied to the training
# matrix (guards against feature-name / ordering bugs between training and deployment)
from ethpredict.live.predictor import Predictor  # noqa: E402
from ethpredict.live.cli import forecast  # noqa: E402

if (MODELS / "meta.json").exists():
    Xfull = pd.concat([X, pd.read_parquet(FEAT / "Xcb.parquet"), pd.read_parquet(FEAT / "Xalt2.parquet")], axis=1)
    for variant in ("full", "spot"):
        p = Predictor(variant, state_file=SEC / "_none.json")
        za, zb = [], []
        for i in idx:
            T = panel["ts"].iloc[i] + pd.Timedelta(minutes=1)
            out, _, _ = forecast(p, src, T)
            direct = p.raw(Xfull.iloc[[i]])
            for key in p.meta["targets"]:
                za.append(out.loc[out.target == key, "z_sat"].values[0]); zb.append(direct[key]["sat"][0])
        za, zb = np.array(za), np.array(zb)
        # live history is 43h instead of 18 months -> 3 slow-EWM vol features differ by ~1e-3, which can
        # flip a tree split; scores must still agree to within a small fraction of their spread
        diff = np.abs(za - zb)
        corr = np.corrcoef(za, zb)[0, 1]
        print(f"predictor parity variant={variant}: n={len(za)} corr={corr:.5f} max|dz|={diff.max():.4f} "
              f"median|dz|={np.median(diff):.5f} (score std {zb.std():.3f})")
        assert np.std(za) > 0, "constant model output -> feature mapping is broken"
        assert corr > 0.99 and diff.max() < 0.25 * zb.std(), "live scores diverge from training scores"
