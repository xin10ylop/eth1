"""Mechanism check 5: WHY would ETH be more (or less) predictable than BTC?  Each hypothesis is a test.

H1 efficiency: return autocorrelation and variance ratios from 1 s to 15 min; permutation entropy
   of 1-min returns (lower = more structure).  Larger deviations from a random walk = more predictable.
H2 leverage cascades: after a large 1-min move (|z| > 3), does the next 1/5 min continue (momentum)
   or revert?  ETH carries more retail leverage -> expect stronger continuation.
H3 slow lead-lag: does BTC lead ETH at 1-min / 5-min bars, even though ETH leads at 1 s?
H4 economics: the same IC is worth sigma-times more bp; fees are charged in bp.
All on pre-holdout days only (2025-05-03 .. 2026-04-30), spot and perp, both assets, same code.
    python scripts/research/mechanisms/why_eth.py
"""
import itertools
import sys
from math import factorial

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from ethpredict.config import RESEARCH, SEC  # noqa: E402

VEN = {"BTC_spot": "spot", "BTC_perp": "um", "ETH_spot": "eth", "ETH_perp": "ethum"}
days = pd.date_range("2025-05-03", "2026-04-30").strftime("%Y-%m-%d")
STEPS = (1, 5, 10, 30, 60, 300, 900)


def perm_entropy(x, m=3):
    pats = {p: i for i, p in enumerate(itertools.permutations(range(m)))}
    w = np.lib.stride_tricks.sliding_window_view(x, m)
    codes = np.array([pats[tuple(np.argsort(r, kind="stable"))] for r in w])
    p = np.bincount(codes, minlength=len(pats)) / len(codes)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(factorial(m)))


rows = []
for day in days:
    lp = {}
    for k, d in VEN.items():
        f = SEC / d / f"{day}.parquet"
        if not f.exists():
            break
        lp[k] = np.log(pd.read_parquet(f, columns=["c"])["c"].ffill().bfill().to_numpy())
    if len(lp) < 4:
        continue
    rec = {"day": day}
    for k, x in lp.items():
        r1 = np.diff(x)
        v1 = r1.var()
        for s in STEPS:
            r = np.diff(x[::s])
            rec[f"{k}_ac{s}"] = np.corrcoef(r[:-1], r[1:])[0, 1]
            if s > 1:
                rec[f"{k}_vr{s}"] = r.var() / (s * v1)
        m1 = np.diff(x[::60])
        rec[f"{k}_pe1m"] = perm_entropy(m1, 3)
        # H2: continuation after large 1-min moves (z vs trailing 60-min sd)
        sd = pd.Series(m1).rolling(60, min_periods=30).std().shift(1).to_numpy()
        z = m1 / sd
        nxt1 = np.r_[m1[1:], np.nan]
        nxt5 = pd.Series(m1[::-1]).rolling(5).sum().to_numpy()[::-1]
        nxt5 = np.r_[nxt5[1:], np.nan]
        big = np.abs(z) > 3
        rec[f"{k}_nbig"] = int(np.nansum(big))
        rec[f"{k}_cont1_sum"] = float(np.nansum(np.sign(z[big]) * nxt1[big] / sd[big]))
        rec[f"{k}_cont5_sum"] = float(np.nansum(np.sign(z[big]) * nxt5[big] / sd[big]))
    # H3: lead-lag at 1 s / 1 min / 5 min bars (spot)
    for s in (1, 60, 300):
        b, e = np.diff(lp["BTC_spot"][::s]), np.diff(lp["ETH_spot"][::s])
        rec[f"btc_leads_eth_{s}"] = np.corrcoef(b[:-1], e[1:])[0, 1]
        rec[f"eth_leads_btc_{s}"] = np.corrcoef(e[:-1], b[1:])[0, 1]
    rows.append(rec)
D = pd.DataFrame(rows)
D.to_csv(RESEARCH / "why_eth_daily.csv", index=False)
D["month"] = D["day"].str[:7]
M = D.groupby("month").mean(numeric_only=True)
pd.set_option("display.width", 200)
print(f"{len(D)} days\n\nH1  return autocorrelation (lag 1) at each sampling step, mean over days  [months ETH|ac| > BTC|ac|]")
for v in ("spot", "perp"):
    for s in STEPS:
        b, e = D[f"BTC_{v}_ac{s}"].mean(), D[f"ETH_{v}_ac{s}"].mean()
        n = int((M[f"ETH_{v}_ac{s}"].abs() > M[f"BTC_{v}_ac{s}"].abs()).sum())
        print(f"  {v:4s} {s:4d}s  BTC {b:+.4f}  ETH {e:+.4f}   [{n}/{len(M)}]")
print("\nH1  variance ratio VR(q) = Var(q-s return) / (q * Var(1-s return))   (1 = random walk)")
for v in ("spot", "perp"):
    for s in STEPS[1:]:
        b, e = D[f"BTC_{v}_vr{s}"].mean(), D[f"ETH_{v}_vr{s}"].mean()
        print(f"  {v:4s} q={s:4d}  BTC {b:.3f}  ETH {e:.3f}")
print("\nH1  permutation entropy of 1-min returns (order 3; 1 = pure noise)")
for v in ("spot", "perp"):
    b, e = D[f"BTC_{v}_pe1m"].mean(), D[f"ETH_{v}_pe1m"].mean()
    print(f"  {v:4s}  BTC {b:.5f}  ETH {e:.5f}   months ETH lower: {int((M[f'ETH_{v}_pe1m'] < M[f'BTC_{v}_pe1m']).sum())}/{len(M)}")
print("\nH2  after a >3 sigma 1-min move: mean signed next move in sigma units (+ = continuation)")
for v in ("spot", "perp"):
    for a in ("BTC", "ETH"):
        n = D[f"{a}_{v}_nbig"].sum()
        print(f"  {a} {v:4s}  n={n:6d}  next 1m {D[f'{a}_{v}_cont1_sum'].sum() / n:+.3f}  next 5m {D[f'{a}_{v}_cont5_sum'].sum() / n:+.3f}")
print("\nH3  lead-lag correlation between BTC and ETH spot returns at bar size s")
for s in (1, 60, 300):
    b, e = D[f"btc_leads_eth_{s}"], D[f"eth_leads_btc_{s}"]
    print(f"  {s:4d}s  BTC->ETH {b.mean():+.4f}   ETH->BTC {e.mean():+.4f}   days BTC leads more: {int((b > e).sum())}/{len(D)}")
