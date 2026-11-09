"""Mechanism check 6: is ETH predictable WHEN BTC MOVES FIRST (BTC-specific news), even if BTC does not
lead on average?  Event: BTC perp moves > k sigma in 10 s while ETH perp moved less than half of
beta * that move ("ETH has not followed yet").  Measure ETH spot's next 10/30/60 s move in the BTC
direction, against the mirror case (ETH moved first, BTC has not followed).  Pre-holdout days.
    python scripts/research/mechanisms/btc_first_events.py
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from ethpredict.config import SEC  # noqa: E402

days = pd.date_range("2025-05-03", "2026-04-30", freq="3D").strftime("%Y-%m-%d")


def lp(d, day):
    return np.log(pd.read_parquet(SEC / d / f"{day}.parquet", columns=["c"])["c"].ffill().bfill().to_numpy())


res = {k: [] for k in ("btc_first", "eth_first")}
for day in days:
    bf, ef, bs, es = lp("um", day), lp("ethum", day), lp("spot", day), lp("eth", day)
    t = np.arange(3600, 86400 - 61)
    rb, re_ = bf[t] - bf[t - 10], ef[t] - ef[t - 10]
    sb = pd.Series(np.diff(bf, prepend=bf[0])).rolling(3600).std().to_numpy()[t] * np.sqrt(10)
    se = pd.Series(np.diff(ef, prepend=ef[0])).rolling(3600).std().to_numpy()[t] * np.sqrt(10)
    for name, lead, lag, sl, slg, tgt in (("btc_first", rb, re_, sb, se, es), ("eth_first", re_, rb, se, sb, bs)):
        zl, zg = lead / sl, lag / slg
        ev = (np.abs(zl) > 4) & (np.sign(zg) * np.abs(zg) * np.sign(zl) < 0.5 * np.abs(zl) * 0.8)
        idx = np.where(ev)[0]
        idx = idx[np.r_[True, np.diff(idx) > 30]] if len(idx) else idx      # one event per 30 s
        for i in idx:
            T = t[i]
            res[name].append({f"next{H}": np.sign(zl[i]) * (tgt[T + H] - tgt[T]) * 1e4 for H in (10, 30, 60)})
for k, v in res.items():
    d = pd.DataFrame(v)
    who = "ETH spot after BTC moved first" if k == "btc_first" else "BTC spot after ETH moved first"
    print(f"{who:34s} n={len(d):5d}  " + "  ".join(
        f"+{H}s: {d[f'next{H}'].mean():+.2f} bp (follows {np.mean(d[f'next{H}'] > 0):.0%})" for H in (10, 30, 60)))
