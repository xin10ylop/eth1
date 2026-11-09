"""Mechanism check 2: is ETH more predictable than BTC?  Apples-to-apples on the same pre-holdout days.

For each asset (target spot), per day: staleness (share of 1 s bars with no price change), trades/s,
1-min volatility, and the IC of three simple, mechanism-based signals for the next 10 s / 60 s:
  perp_gap  = log perp - log spot (perp leads spot)
  mom10_f   = perp return over the last 10 s
  cross10   = the OTHER asset's perp return over the last 10 s
    python scripts/research/mechanisms/predictability_compare.py
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from ethpredict.config import SEC  # noqa: E402

DAYS = [f"2025-{m:02d}-{d:02d}" for m in range(6, 13) for d in (5, 12, 18, 25)] + \
       [f"2026-{m:02d}-{d:02d}" for m in range(1, 5) for d in (5, 12, 18, 25)]
A = {"BTC": ("spot", "um"), "ETH": ("eth", "ethum")}


def rd(d, day, cols=("c",)):
    return pd.read_parquet(SEC / d / f"{day}.parquet", columns=list(cols))


rows = []
for day in DAYS:
    px = {}
    for a, (ds, df) in A.items():
        s = rd(ds, day, ("c", "n")); f = rd(df, day, ("c", "n"))
        px[a] = (np.log(s["c"].ffill().bfill().to_numpy()), np.log(f["c"].ffill().bfill().to_numpy()),
                 s["n"].to_numpy(), f["n"].to_numpy())
    T = np.arange(900, 86400 - 61, 60)
    for a, other in (("BTC", "ETH"), ("ETH", "BTC")):
        ls, lf, ns, nf = px[a]
        lo = px[other][1]
        r1 = np.diff(ls)
        sig = np.std(ls[T] - ls[T - 60])
        rec = dict(day=day, asset=a, stale_spot=np.mean(r1 == 0), stale_perp=np.mean(np.diff(lf) == 0),
                   n_spot=ns.mean(), n_perp=nf.mean(), sigma1m_bp=1e4 * sig)
        for H in (10, 60):
            y = ls[T + H - 1] - ls[T - 1]          # P(T) = close of second T-1
            e = T - 1
            sigs = {"perp_gap": lf[e] - ls[e], "mom10_f": lf[e] - lf[e - 10], "mom10_s": ls[e] - ls[e - 10],
                    "cross10": lo[e] - lo[e - 10]}
            for k, v in sigs.items():
                rec[f"ic{H}_{k}"] = np.corrcoef(v, y)[0, 1]
        rows.append(rec)
D = pd.DataFrame(rows)
D["month"] = D["day"].str[:7]
pd.set_option("display.width", 220)
print(D.groupby("asset").mean(numeric_only=True).T.round(4).to_string())
print("\nmonths where ETH IC > BTC IC (monthly means):")
m = D.groupby(["month", "asset"]).mean(numeric_only=True)
for c in [c for c in D if c.startswith("ic")]:
    x = m[c].unstack()
    print(f"  {c:16s} ETH>BTC in {int((x['ETH'] > x['BTC']).sum())}/{len(x)} months   mean diff {float((x['ETH'] - x['BTC']).mean()):+.4f}")
