"""Charts + summary tables from out-of-sample predictions.

python scripts/reports/make_report.py --pattern "final_full_{key}_holdout" --keys s1,f1,s5,f5 --spot_pattern "final_spot_{key}_holdout"
"""
import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.simplefilter("ignore")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ethpredict.modeling import evaluate as E  # noqa: E402
from ethpredict.config import FEAT, FIGURES, HOLDOUT, PREDS  # noqa: E402

# reference palette (dataviz skill), light mode
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "lines.linewidth": 2})

ap = argparse.ArgumentParser()
ap.add_argument("--pattern", default="final_full_{key}_holdout")
ap.add_argument("--keys", default="s1,f1,s5,f5")
ap.add_argument("--tag", default="holdout")
a = ap.parse_args()
keys = a.keys.split(",")
LABEL = {"s10s": "spot +10s", "f10s": "perp +10s", "s15s": "spot +15s", "f15s": "perp +15s", "s1": "spot +1m", "f1": "perp +1m", "s5": "spot +5m", "f5": "perp +5m", "s2": "spot +2m", "s3": "spot +3m",
         "s4": "spot +4m", "f2": "perp +2m", "f3": "perp +3m", "f4": "perp +4m"}

ycols = ["ts", "sigma"] + [f"y_{v}{h}" for v in "sf" for h in range(1, 6)] + \
        [f"y_{v}{h}_d{d}" for v in "sf" for h in range(1, 6) for d in (1, 2, 5, 10, 30)]
df = pd.read_parquet(FEAT / "panel.parquet", columns=ycols)
df = df[df["ts"] >= "2025-05-03"].reset_index(drop=True)
if (FEAT / "Ysub.parquet").exists():
    df = pd.concat([df, pd.read_parquet(FEAT / "Ysub.parquet")], axis=1)
df = df.set_index("ts")
P = {}
for k in keys:
    f = PREDS / (a.pattern.format(key=k) + ".parquet")
    if f.exists():
        P[k] = pd.read_parquet(f).set_index("ts")["pred"]
summary = {}

# ---------- latency decay
delays = [0, 1, 2, 5, 10, 30]
lat = {}
for k, p in P.items():
    v, h = k[0], k[1:]
    d = df.loc[p.index]
    dl_ok = [dl for dl in delays if dl == 0 or f"y_{v}{h}_d{dl}" in d]
    lat[k] = [E.metrics((d[f"y_{v}{h}"] if dl == 0 else d[f"y_{v}{h}_d{dl}"]).values, p.values)["ic"] if dl in dl_ok else np.nan
              for dl in delays]
show = [k for k in keys if k in lat][:3]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for i, k in enumerate(show):
    ax.plot(delays, lat[k], marker="o", ms=5, color=SERIES[i], label=LABEL[k])
ax.set_xscale("symlog", linthresh=1)
dmax = max(d for d in delays if any(np.isfinite(lat[k][delays.index(d)]) for k in show))
ax.set_xticks([d for d in delays if d <= dmax], [str(d) for d in delays if d <= dmax])
ax.set_xlabel("execution delay after bar close (seconds)")
ax.set_ylabel("information coefficient")
ax.set_title("Second-horizon forecasts: value vs delivery delay" if all(k.endswith("s") for k in show)
             else "The 1-minute edge decays within seconds")
ax.set_ylim(bottom=0)
ax.legend(frameon=False, loc="upper right")
ax.set_xlim(-0.2, dmax * 1.3)
fig.tight_layout()
fig.savefig(FIGURES / f"latency_decay_{a.tag}.png", dpi=150)
summary["latency_ic"] = {k: dict(zip(map(str, delays), np.round(v, 4))) for k, v in lat.items()}

# ---------- hit rate by confidence decile
fig, ax = plt.subplots(figsize=(6.4, 3.6))
w = 0.26
hits = {}
for i, k in enumerate(show):
    p = P[k]; v, h = k[0], k[1:]
    y = df.loc[p.index, f"y_{v}{h}"].values
    m = np.isfinite(y) & (y != 0)
    dec = pd.qcut(np.abs(p.values[m]), 10, labels=False)
    hr = pd.Series(np.sign(p.values[m]) == np.sign(y[m])).groupby(dec).mean()
    hits[k] = hr.round(4).tolist()
    ax.bar(np.arange(10) + (i - 1) * w, hr.values * 100, width=w - 0.03, color=SERIES[i], label=LABEL[k])
ax.axhline(50, color=INK2, lw=1, ls="--")
ax.set_xticks(range(10), [f"{10*i}-{10*(i+1)}" for i in range(10)], fontsize=8)
ax.set_xlabel("confidence decile (|forecast| percentile)")
ax.set_ylabel("directional hit rate (%)")
ax.set_ylim(45, max(70, max(max(v) for v in hits.values()) * 100 + 2))
ax.set_title("Hit rate rises with model confidence")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIGURES / f"hit_by_confidence_{a.tag}.png", dpi=150)
summary["hit_by_conf_decile"] = hits

# ---------- monthly IC
mon = {}
for k, p in P.items():
    v, h = k[0], k[1:]
    mon[k] = E.monthly_ic(pd.Series(p.index), df.loc[p.index, f"y_{v}{h}"].values, p.values).round(4)
summary["monthly_ic"] = {k: v.to_dict() for k, v in mon.items()}

# ---------- when is ETH predictable? (spot 1m and perp 5m)
regimes = {}
for k in [k for k in keys if k in P]:
    p = P[k]; v, h = k[0], k[1:]
    d = pd.DataFrame({"p": p.values, "y": df.loc[p.index, f"y_{v}{h}"].values, "sig": df.loc[p.index, "sigma"].values}, index=p.index).dropna()
    hr = (d.index + pd.Timedelta(minutes=1)).hour
    sess = np.select([hr < 8, hr < 13, hr < 21], ["Asia 00-08", "Europe 08-13", "US 13-21"], "Late US 21-24")
    wk = np.where(d.index.dayofweek >= 5, "weekend", "weekday")
    vt = pd.qcut(d["sig"], 3, labels=["low vol", "mid vol", "high vol"]).astype(str)
    r = {}
    for name, g in (("session", sess), ("week", wk), ("vol", vt)):
        r[name] = d.groupby(g).apply(lambda x: pd.Series({"ic": np.corrcoef(x.p, x.y)[0, 1],
                                                           "hit": np.mean(np.sign(x.p[x.y != 0]) == np.sign(x.y[x.y != 0]))}),
                                     include_groups=False).round(4).to_dict("index")
    regimes[k] = r
summary["regimes"] = regimes
(HOLDOUT / f"summary_{a.tag}.json").write_text(json.dumps(summary, indent=1, default=float))
print(json.dumps(summary, indent=1, default=float)[:4000])
