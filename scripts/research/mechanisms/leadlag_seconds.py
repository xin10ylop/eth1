"""Mechanism check 1: who leads whom at the 1-second scale (pre-holdout days only).

For BTC spot/perp and ETH spot/perp, cross-correlation of 1 s log returns at lags -5..+10 s, and the
share of ETH's next-60 s move explained by moves that already happened elsewhere ("catch-up").
    python scripts/research/mechanisms/leadlag_seconds.py
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from ethpredict.config import SEC, DIR_SPOT, DIR_PERP, DIR_LEAD_SPOT, DIR_LEAD_PERP, RESEARCH  # noqa: E402

DAYS = [f"2025-{m:02d}-{d:02d}" for m in range(6, 13) for d in (5, 18)] + \
       [f"2026-{m:02d}-{d:02d}" for m in range(1, 5) for d in (5, 18)]       # 22 pre-holdout days
VEN = {"eth_s": DIR_SPOT, "eth_f": DIR_PERP, "btc_s": DIR_LEAD_SPOT, "btc_f": DIR_LEAD_PERP}


def load(day):
    out = {}
    for k, d in VEN.items():
        c = pd.read_parquet(SEC / d / f"{day}.parquet", columns=["c"])["c"].ffill().bfill().to_numpy()
        out[k] = np.diff(np.log(c), prepend=np.log(c[0]))
    return out


def xcorr(a, b, lag):   # corr(a_t, b_{t+lag}); lag>0: a leads b
    if lag > 0:
        a, b = a[:-lag], b[lag:]
    elif lag < 0:
        a, b = a[-lag:], b[:lag]
    return np.corrcoef(a, b)[0, 1]


rows, pred = [], []
for day in DAYS:
    r = load(day)
    for a, b in (("btc_f", "eth_f"), ("btc_f", "eth_s"), ("btc_s", "eth_s"), ("eth_f", "eth_s"), ("btc_f", "btc_s"),
                 ("eth_f", "btc_f")):
        for lag in range(-3, 11):
            rows.append(dict(day=day, pair=f"{a}->{b}", lag=lag, corr=xcorr(r[a], r[b], lag)))
    # catch-up regression at minute boundaries: ETH spot next 60 s vs past 10 s moves
    cs = {k: np.cumsum(v) for k, v in r.items()}
    T = np.arange(600, 86400 - 61, 60)
    y = cs["eth_s"][T + 60] - cs["eth_s"][T]
    past = {k: cs[k][T] - cs[k][T - 10] for k in cs}
    pred.append(pd.DataFrame(dict(day=day, y=y, **{f"p10_{k}": v for k, v in past.items()})))

cc = pd.DataFrame(rows).groupby(["pair", "lag"])["corr"].mean().unstack()
pd.set_option("display.width", 200)
print("mean cross-correlation of 1 s returns, corr(a_t, b_{t+lag}) (lag>0: a leads b)")
print(cc.round(3).to_string())
P = pd.concat(pred)
print("\nIC of past-10 s moves for ETH spot next-60 s return (per-day mean, pooled):")
for c in [c for c in P if c.startswith("p10_")]:
    ic = P.groupby("day").apply(lambda g: np.corrcoef(g[c], g["y"])[0, 1])
    print(f"  {c:10s} mean IC {ic.mean():+.4f}  days>0 {int((ic > 0).sum())}/{len(ic)}")
# relative move: BTC-perp minus beta*ETH-perp (the 'unfinished' part of a market-wide move)
g = P.copy()
b = (g["p10_btc_f"] * g["p10_eth_f"]).sum() / (g["p10_btc_f"] ** 2).sum()
g["gap"] = g["p10_btc_f"] * (1.0 / max(b, 1e-9)) * 0 + g["p10_btc_f"] - g["p10_eth_f"] / max(b, 1e-9)
ic = g.groupby("day").apply(lambda d: np.corrcoef(d["gap"], d["y"])[0, 1])
print(f"  gap = btc_f - eth_f/beta (beta={b:.2f}) mean IC {ic.mean():+.4f}  days>0 {int((ic > 0).sum())}/{len(ic)}")
cc.to_csv(RESEARCH / "leadlag_seconds.csv")
