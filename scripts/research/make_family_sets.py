"""Write research feature sets defined by FAMILY (mechanism), not by hand-picked columns.

    python scripts/research/make_family_sets.py      -> configs/feature_sets/research/*.txt

base  = ETH spot + perp microstructure, basis, perp premium index, Levy areas, minute state, calendar
+btc  = the other major's SPOT (e_*, be_*, E*, BE*, x_beta, levy_es*)           -> 'core'
sat   = core + swarm5 + ETH/USDC book + Coinbase + swarm17 + BTC perp (xp_*)     -> 'sat_all'
loo_* = sat_all minus one family (leave-one-family-out)
"""
import re
import sys

import pandas as pd

sys.path.insert(0, ".")
from ethpredict.config import FEAT, FEATURE_SETS  # noqa: E402
from ethpredict.features import matrix as F  # noqa: E402

OUT = FEATURE_SETS / "research"
OUT.mkdir(parents=True, exist_ok=True)
FAMILIES = {
    "cb": r"^cb_", "xperp": r"^(xp_|x_basis)", "swarm17": r"^swarm2_", "swarm5": r"^alt_", "usdc": r"^book_",
    "btcspot": r"^(e_|be_|E\d|BE\d|levy_es|x_beta|x_resid)", "premium": r"^prem",
}
BTC_CORE = open("/home/user/btc-predict/configs/feature_sets/core.txt").read().split() if len(sys.argv) < 2 else []


def fam(c):
    for k, p in FAMILIES.items():
        if re.search(p, c):
            return k
    return "base"


panel = pd.read_parquet(FEAT / "panel.parquet").head(5000)
cols = list(pd.read_parquet(FEAT / "X.parquet").columns) + list(F.extra_features(panel).columns)
for f in ("Xalt", "Xalt2", "Xcb"):
    cols += list(pd.read_parquet(FEAT / f"{f}.parquet").columns)
cols = list(dict.fromkeys(cols))
# start from the columns the BTC 'core' used (same feature code), so 'base' excludes research-only aux
# (depth/OI do not exist for ETH) and the few columns BTC pruned for being degenerate
rename = {"eth_beta": "x_beta", "eth_resid5": "x_resid5"}
core_tpl = {rename.get(c, c) for c in BTC_CORE}
base = [c for c in cols if fam(c) == "base" and (c in core_tpl or c == "f_avgsize")]
by = {k: [c for c in cols if fam(c) == k] for k in FAMILIES}
sets = {"base": base + by["premium"], "core": base + by["premium"] + by["btcspot"]}
sets["sat_all"] = sets["core"] + by["swarm5"] + by["usdc"] + by["cb"] + by["swarm17"] + by["xperp"]
for k in ("cb", "xperp", "swarm17", "swarm5", "usdc", "btcspot", "premium"):
    sets[f"loo_{k}"] = [c for c in sets["sat_all"] if fam(c) != k]
# lean satellites (simplest sets that keep the families with evidence)
sets["lean_cb_xp"] = sets["core"] + by["cb"] + by["xperp"]
sets["lean_cb_xp_s5"] = sets["core"] + by["cb"] + by["xperp"] + by["swarm5"]
sets["noswarm"] = [c for c in sets["sat_all"] if fam(c) not in ("swarm5", "swarm17")]
# 'fast' = minus slow regime features (24h returns, hour-of-day, round numbers...): a property of the
# feature, not of the asset, so the list is the one btc-predict derived (core minus core_fast)
bc = open("/home/user/btc-predict/configs/feature_sets/core.txt").read().split()
bf = set(open("/home/user/btc-predict/configs/feature_sets/core_fast.txt").read().split())
SLOW = {rename.get(c, c) for c in bc if c not in bf}
for k in ("core", "sat_all", "lean_cb_xp", "lean_cb_xp_s5"):
    sets[f"{k}_fast"] = [c for c in sets[k] if c not in SLOW]
for k, v in sets.items():
    (OUT / f"{k}.txt").write_text("\n".join(v) + "\n")
    print(f"{k:14s} {len(v)}")

# ---- PRODUCTION sets (decided on DEV months, see docs/FINDINGS.md):
#   core = own ETH spot+perp market + perp premium + BTC spot;  sat = core + Coinbase + BTC perp + 5-coin swarm
#   spot variant = the same minus everything that needs Binance futures data
bs = open("/home/user/btc-predict/configs/feature_sets/spotcore.txt").read().split()
FUT = ({rename.get(c, c) for c in bc if c not in set(bs)} | set(by["xperp"]) | set(by["premium"])
       | {c for c in cols if c.startswith(("f_", "basis", "levy_f", "rv_f_"))})
prod = {"core": sets["core"], "sat": sets["lean_cb_xp_s5"]}
prod["spotcore"] = [c for c in prod["core"] if c not in FUT]
prod["spotsat"] = [c for c in prod["sat"] if c not in FUT]
for k in list(prod):
    prod[k + "_fast"] = [c for c in prod[k] if c not in SLOW]
for k, v in prod.items():
    (FEATURE_SETS / f"{k}.txt").write_text("\n".join(v) + "\n")
    print(f"production {k:14s} {len(v)}")
print({k: len(v) for k, v in by.items()})
