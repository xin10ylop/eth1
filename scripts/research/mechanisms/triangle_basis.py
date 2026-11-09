"""Mechanism check 3: does the ETH/BTC ratio book lead ETHUSDT through triangular arbitrage?

tri_gap = log(ETHBTC) - [log(ETHUSDT) - log(BTCUSDT)]: >0 means the ratio book prices ETH richer than
the USDT books imply -> arbitrage should lift ETHUSDT (and/or lower BTCUSDT, ETHBTC).
Also: the 1 s lead-lag of the ratio book vs the implied ratio. Pre-holdout days only.
    python scripts/research/mechanisms/triangle_basis.py
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from ethpredict.config import SEC  # noqa: E402

DAYS = [f"2025-{m:02d}-{d:02d}" for m in range(6, 13) for d in (5, 12, 18, 25)] + \
       [f"2026-{m:02d}-{d:02d}" for m in range(1, 5) for d in (5, 12, 18, 25)]


def lc(d, day):
    x = pd.read_parquet(SEC / d / f"{day}.parquet", columns=["c", "v", "n"])
    return np.log(x["c"].ffill().bfill().to_numpy()), x["n"].to_numpy()


rows = []
for day in DAYS:
    e, _ = lc("eth", day); b, _ = lc("spot", day); r, nr = lc("ethbtc", day); ef, _ = lc("ethum", day)
    implied = e - b
    gap = r - implied
    gap = gap - pd.Series(gap).rolling(3600, min_periods=60).median().bfill().to_numpy()   # remove slow level
    dr, di = np.diff(r), np.diff(implied)
    rec = dict(day=day, ethbtc_trades_per_s=nr.mean(), stale_ratio=np.mean(dr == 0),
               ratio_leads_implied=np.corrcoef(dr[:-1], di[1:])[0, 1], implied_leads_ratio=np.corrcoef(di[:-1], dr[1:])[0, 1],
               gap_bp_std=1e4 * np.nanstd(gap))
    T = np.arange(3700, 86400 - 61, 60); t = T - 1
    for H in (10, 60):
        ye = e[t + H] - e[t]; yb = b[t + H] - b[t]; yr = r[t + H] - r[t]
        rec[f"ic{H}_gap->eth"] = np.corrcoef(gap[t], ye)[0, 1]
        rec[f"ic{H}_gap->btc"] = np.corrcoef(gap[t], yb)[0, 1]
        rec[f"ic{H}_gap->ratio"] = np.corrcoef(gap[t], yr)[0, 1]
        # partial: does the gap add to the perp-spot gap for ETH?
        X = np.c_[np.ones(len(t)), ef[t] - e[t], gap[t]]
        beta = np.linalg.lstsq(X, ye, rcond=None)[0]
        rec[f"t{H}_gap_given_basis"] = beta[2] / (np.std(ye - X @ beta) / np.sqrt(len(t)) / np.std(gap[t]))
    rows.append(rec)
D = pd.DataFrame(rows); D["month"] = D["day"].str[:7]
pd.set_option("display.width", 200)
print(D.mean(numeric_only=True).round(4).to_string())
m = D.groupby("month").mean(numeric_only=True)
for c in ("ic10_gap->eth", "ic60_gap->eth", "ic60_gap->ratio"):
    print(f"{c:18s} months>0: {int((m[c] > 0).sum())}/{len(m)}  monthly: {' '.join(f'{v:+.3f}' for v in m[c])}")
