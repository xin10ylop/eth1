"""Markdown results table from reports/{holdout,confirm}/<kind>_<variant>_<key>.json

python scripts/reports/results_table.py holdout
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from ethpredict.config import CONFIRM, HOLDOUT  # noqa: E402

kind = sys.argv[1] if len(sys.argv) > 1 else "holdout"
rows = []
for f in sorted((HOLDOUT if kind == "holdout" else CONFIRM).glob(f"{kind}_*_*.json")):
    variant, key = f.stem.split("_")[1], f.stem.split("_")[2]
    r = json.loads(f.read_text())
    e = r["ens"]
    hz = key[1:]
    order = int(hz[:-1]) / 60 if hz.endswith("s") else int(hz)
    name = ("spot" if key[0] == "s" else "perp") + (f" +{hz}" if hz.endswith("s") else f" +{hz}m")
    rows.append((variant, order, key[0], name, e, r))
rows.sort(key=lambda x: (x[0] != "full", x[2] != "s", x[1]))
print("| model | target | hit rate | AUC | IC | top-10% conf. hit | top-1% conf. hit | log-loss skill | hit @ +2s | hit @ +5s | IC core → ens |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for variant, h, v, name, e, r in rows:
    print(f"| {variant} | {name} | **{100*e['hit']:.1f}%** | {e['auc']:.3f} | {e['ic']:.3f} | {100*e['hit_top10']:.1f}% | "
          f"{100*e['hit_top1']:.1f}% | {r['ens_calibration']['logloss_skill_pct']:.2f}% | {100*e['hit_d2']:.1f}% | {100*e['hit_d5']:.1f}% | "
          f"{r['core']['ic']:.3f} → {e['ic']:.3f} |")
print()
print("Monthly IC (ensemble):")
for variant, h, v, name, e, r in rows:
    print(f"* {variant} {name}: " + ", ".join(f"{k} {val:+.3f}" for k, val in e["monthly_ic"].items()))
