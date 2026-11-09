"""Score logged live forecasts against realized prices (Binance spot 1m klines; perp if reachable)."""
import json
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, ".")
from ethpredict.config import LOGS, SPOT_SYMBOL  # noqa: E402

rows = [json.loads(l) for l in open(LOGS / "predictions.jsonl")]
recs = []
for r in rows:
    for f in r["forecasts"]:
        recs.append({"T": pd.Timestamp(r["T"]), "spot": r["spot"], "perp": r["perp"], **f})
d = pd.DataFrame(recs)
t0 = int(d["T"].min().value // 10**6)
t1 = int((d["T"].max() + pd.Timedelta(minutes=6)).value // 10**6)
k = []
s = t0
while s < t1:
    x = requests.get("https://data-api.binance.vision/api/v3/klines",
                     params={"symbol": SPOT_SYMBOL, "interval": "1m", "startTime": s, "endTime": t1, "limit": 1000}, timeout=10).json()
    if not x:
        break
    k += x
    s = x[-1][0] + 60000
close = pd.Series({pd.Timestamp(int(x[0]), unit="ms") + pd.Timedelta(minutes=1): float(x[4]) for x in k})
# 1-second closes for sub-minute horizons: P(t) = close of second t-1
k1, s1 = [], t0
while s1 < t1:
    x = requests.get("https://data-api.binance.vision/api/v3/klines",
                     params={"symbol": SPOT_SYMBOL, "interval": "1s", "startTime": s1, "endTime": t1, "limit": 1000}, timeout=10).json()
    if not x:
        break
    k1 += x
    s1 = x[-1][0] + 1000
sec = pd.Series({pd.Timestamp(int(x[0]), unit="ms") + pd.Timedelta(seconds=1): float(x[4]) for x in k1}).sort_index()
d = d[d["venue"] == "s"].copy()


def realized(T, h, p):
    t_end = T + pd.Timedelta(minutes=h)
    if h >= 1:
        return 1e4 * np.log(close.get(t_end, np.nan) / p)
    a, b = sec.asof(T), sec.asof(t_end)
    return 1e4 * np.log(b / a)


d["realized_bp"] = [realized(T, h, p) for T, h, p in zip(d["T"], d["h"], d["spot"])]
d["h"] = [(f"{round(h * 60)}s" if h < 1 else f"{int(h)}m") for h in d["h"]]
d = d.dropna(subset=["realized_bp"])
d = d[d["realized_bp"] != 0]
def summ(g):
    n = len(g)
    hit = np.mean(np.sign(g.exp_ret_bp) == np.sign(g.realized_bp))
    a = g.z.abs()
    top = g[a >= a.median()]
    ic = np.corrcoef(g.exp_ret_bp, g.realized_bp)[0, 1] if n > 2 else np.nan
    return pd.Series({"n": n, "hit": hit, "hit_se": np.sqrt(0.25 / n),
                      "hit_top50%conf": np.mean(np.sign(top.exp_ret_bp) == np.sign(top.realized_bp)), "n_top": len(top),
                      "ic": ic, "ic_se": 1 / np.sqrt(max(n - 3, 1)),
                      "in_80band": np.mean((g.realized_bp >= g.lo80_bp) & (g.realized_bp <= g.hi80_bp))})


print(d.groupby("h").apply(summ, include_groups=False).round(3).to_string())
print("note: consecutive forecasts share market conditions; treat the standard errors as optimistic.")
