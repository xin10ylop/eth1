"""ETH vs BTC on their frozen holdouts (same months, same machinery): hit rate and bp edge by horizon.

    python scripts/reports/compare_btc.py [/path/to/btc-predict/reports/holdout]
Writes reports/figures/eth_vs_btc.png and reports/holdout/eth_vs_btc.json.
Two panels (never a dual axis): directional hit rate, and edge in bp (= mean of sign(forecast) x move).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
from ethpredict.config import FIGURES, HOLDOUT  # noqa: E402

BTC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../btc-predict/reports/holdout")
KEYS = ["10s", "15s", "1", "2", "3", "4", "5"]
LAB = ["10 s", "15 s", "1 m", "2 m", "3 m", "4 m", "5 m"]
ETH_C, BTC_C = "#2a78d6", "#eb6834"      # categorical slots 1, 2 (validated: CVD dE 24.7, contrast >= 3:1)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(folder, venue):
    out = {}
    for k in KEYS:
        f = folder / f"holdout_full_{venue}{k}.json"
        if f.exists():
            r = json.loads(f.read_text())["ens"]
            out[k] = {m: r[m] for m in ("ic", "rank_ic", "hit", "hit_top10", "edge_bps", "edge_bps_top10", "ic_d2")}
    return out


res = {a: {v: load(p, v) for v in ("s", "f")} for a, p in (("ETH", HOLDOUT), ("BTC", BTC))}
(HOLDOUT / "eth_vs_btc.json").write_text(json.dumps(res, indent=1))

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor=SURF)
w = 0.38
for col, venue, vname in ((0, "s", "spot"), (1, "f", "perpetual")):
    for row, metric, title, fmt in ((0, "hit", "Directional hit rate", "{:.1%}"), (1, "edge_bps", "Edge per forecast (bp)", "{:.2f}")):
        ax = axes[row, col]
        ax.set_facecolor(SURF)
        for i, (asset, c) in enumerate((("ETH", ETH_C), ("BTC", BTC_C))):
            d = res[asset][venue]
            ks = [k for k in KEYS if k in d]
            if not ks:
                continue
            xs = [j + (i - 0.5) * w for j, k in enumerate(KEYS) if k in d]
            ys = [d[k][metric] - (0.5 if metric == "hit" else 0) for k in ks]
            ax.bar(xs, ys, width=w - 0.04, color=c, label=asset, edgecolor=SURF, linewidth=2, zorder=3)
            # selective direct labels: ETH (the subject) only, first and last horizon
            for x, y, k in ([(xs[0], ys[0], ks[0]), (xs[-1], ys[-1], ks[-1])] if asset == "ETH" else []):
                v = d[k][metric]
                ax.text(x, y, fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(range(len(KEYS)), LAB)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(length=0)
        if metric == "hit":
            ax.set_yticks(ax.get_yticks(), [f"{50 + 100 * t:.0f}%" for t in ax.get_yticks()])
        ax.set_title(f"{title}, {vname}", loc="left", color=INK, fontsize=11)
axes[0, 0].legend(frameon=False, loc="upper right")
fig.suptitle("ETH vs BTC on the same frozen holdout (May–Oct 2026, full model, ensemble)", x=0.01, ha="left",
             color=INK, fontsize=12)
fig.text(0.01, 0.005, "Hit rate is drawn from 50%. Edge = mean of sign(forecast) × realized move, before costs. "
         "ETH moved ~1.35× more than BTC per minute here: enough to tie at 10-15 s, not beyond.",
         color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
fig.savefig(FIGURES / "eth_vs_btc.png", dpi=150, facecolor=SURF)
for v in ("s", "f"):
    for k in KEYS:
        e, b = res["ETH"][v].get(k), res["BTC"][v].get(k)
        if e and b:
            print(f"{v}{k:4s} IC ETH {e['ic']:.4f} BTC {b['ic']:.4f} | hit {e['hit']:.4f} {b['hit']:.4f} | "
                  f"edge bp {e['edge_bps']:.3f} {b['edge_bps']:.3f} | top10 hit {e['hit_top10']:.4f} {b['hit_top10']:.4f}")
