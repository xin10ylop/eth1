"""Mechanism check 4: univariate IC of every feature family, ETH vs BTC, on PRE-HOLDOUT months only.

For each feature: Pearson IC with the scale-free target per month (2025-05 .. 2026-04), then per
family: best feature's mean |IC|, its sign stability, and the median |IC|. Run on both repos'
feature matrices (identical feature code) to see what is different about ETH.
    python scripts/research/mechanisms/family_ic_scan.py /home/user/eth1/data/features ETH
    python scripts/research/mechanisms/family_ic_scan.py /home/user/btc-predict/data/features BTC
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from ethpredict.features import matrix as F  # noqa: E402
from ethpredict.config import RESEARCH  # noqa: E402

feat, tag = Path(sys.argv[1]), sys.argv[2]
panel = pd.read_parquet(feat / "panel.parquet", columns=None)
panel = panel[panel["ts"] >= "2025-05-03"].reset_index(drop=True)
keep = panel["ts"] < "2026-05-01"
parts = [pd.read_parquet(feat / "X.parquet"), F.extra_features(panel)]
for f in ("Xalt", "Xalt2", "Xcb"):
    if (feat / f"{f}.parquet").exists():
        parts.append(pd.read_parquet(feat / f"{f}.parquet"))
X = pd.concat(parts, axis=1).loc[keep].reset_index(drop=True)
X = X.loc[:, ~X.columns.duplicated()]
ys = pd.read_parquet(feat / "Ysub.parquet")
month = panel.loc[keep, "ts"].dt.strftime("%Y-%m").reset_index(drop=True)
targets = {"s10s": None, "s1": 1, "s5": 5, "f1": 1}
Y = {}
for k, h in targets.items():
    if k.endswith("s") and k[1:-1].isdigit():
        y = panel.loc[keep, f"y_{k}"] if f"y_{k}" in panel else ys.loc[keep.values, f"y_{k}"]
        sc = F.target_scale(panel, 1).loc[keep] * np.sqrt(int(k[1:-1]) / 60)
    else:
        y = panel.loc[keep, f"y_{k}"]; sc = F.target_scale(panel, h).loc[keep]
    Y[k] = (y / sc).clip(-6, 6).reset_index(drop=True)

FAM = [("cb", r"^cb_"), ("xperp(other major perp)", r"^(xp_|x_basis|ethp_|eth_basis)"), ("swarm17", r"^swarm2_"),
       ("swarm5", r"^alt_"), ("usdc_book", r"^book_"), ("other_major_spot", r"^(e_|be_|E\d|BE\d|levy_es|x_beta|x_resid|eth_beta|eth_resid)"),
       ("basis", r"^(basis_|rv_f_over_s)"), ("premium", r"^prem"), ("levy_own", r"^levy_"), ("perp_micro", r"^f_"),
       ("spot_micro", r"^s_"), ("calendar", r"^(tod_|dow|minute|mod|qh_|hr_|to_funding)"), ("minute_state", r".")]


def fam(c):
    return next(n for n, p in FAM if re.search(p, c))


rows = []
for c in X.columns:
    x = X[c].astype(float)
    if x.std() == 0 or x.notna().mean() < 0.5:
        continue
    r = {"feature": c, "family": fam(c)}
    for k, y in Y.items():
        ok = x.notna() & y.notna()
        d = pd.DataFrame({"x": x[ok], "y": y[ok], "m": month[ok]})
        mic = d.groupby("m").apply(lambda g: np.corrcoef(g["x"], g["y"])[0, 1] if g["x"].std() > 0 else np.nan)
        r[f"{k}_ic"] = mic.mean(); r[f"{k}_pos"] = float(np.mean(np.sign(mic) == np.sign(mic.mean())))
    rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(RESEARCH / f"feature_ic_{tag}.csv", index=False)
out = []
for f, g in R.groupby("family"):
    rec = {"family": f, "n": len(g)}
    for k in Y:
        i = g[f"{k}_ic"].abs().idxmax()
        rec[f"{k}_best"] = f"{g.loc[i, 'feature']} {g.loc[i, f'{k}_ic']:+.3f} ({g.loc[i, f'{k}_pos']:.0%})"
        rec[f"{k}_med|ic|"] = round(g[f"{k}_ic"].abs().median(), 4)
    out.append(rec)
O = pd.DataFrame(out).sort_values("s1_med|ic|", ascending=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
print(f"== {tag}: per-family univariate IC on pre-holdout months (best feature: mean IC, share of months with that sign)")
print(O[["family", "n", "s10s_best", "s1_best", "s1_med|ic|", "s5_best", "f1_best"]].to_string(index=False))
O.to_csv(RESEARCH / f"family_ic_{tag}.csv", index=False)
