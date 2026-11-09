"""Generic walk-forward runner that caches out-of-sample predictions for later stacking.

python scripts/research/run_model.py --model lgbm --venue s --h 1 --months dev --tag base
python scripts/research/run_model.py --venue s --h 10s --months 2025-11,2026-01,2026-03 --tag core --features @configs/...
"""
import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.simplefilter("ignore")
from ethpredict.modeling import evaluate as E  # noqa: E402
from ethpredict.features import matrix as F  # noqa: E402
from ethpredict.modeling import gbm as M  # noqa: E402
from ethpredict.config import DEV, FEAT, PREDS  # noqa: E402

PRED = PREDS
PRED.mkdir(exist_ok=True)

_cache = {}


def load(extra: bool = True, alts: bool = False, more: str = ""):
    if "df" not in _cache:
        df = pd.read_parquet(FEAT / "panel.parquet")
        df = df[df["ts"] >= "2025-05-03"].reset_index(drop=True)
        X = pd.read_parquet(FEAT / "X.parquet")
        if (FEAT / "Ysub.parquet").exists():      # 10 s / 15 s targets
            ys = pd.read_parquet(FEAT / "Ysub.parquet")
            df = pd.concat([df, ys.drop(columns=[c for c in ys if c in df])], axis=1)
        _cache["df"], _cache["X"] = df, X
    df, X = _cache["df"], _cache["X"]
    if extra:
        if "Xe" not in _cache:
            _cache["Xe"] = F.extra_features(df)
        X = pd.concat([X, _cache["Xe"]], axis=1)
    if alts and (FEAT / "Xalt.parquet").exists():
        X = pd.concat([X, pd.read_parquet(FEAT / "Xalt.parquet")], axis=1)
    for f in [x for x in more.split(",") if x]:
        X = pd.concat([X, pd.read_parquet(FEAT / f)], axis=1)
    return df, X


def months_arg(s):
    if s == "dev":
        return E.DEV_MONTHS
    if s == "holdout":
        return E.HOLDOUT_MONTHS
    if s == "all":
        return E.DEV_MONTHS + E.HOLDOUT_MONTHS
    return s.split(",")


def target_series(df, venue, h, delay=None):
    if venue == "basis":   # change of perp-spot basis over horizon (component of spot return)
        b = df["basis"]
        ok = df[f"y_s{h}"].notna()
        return (b.shift(-h) - b).where(ok)
    col = f"y_{venue}{h}" if delay is None else f"y_{venue}{h}_d{delay}"
    return df[col]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--venue", default="s")
    ap.add_argument("--h", default="1", help="minutes (1..5) or seconds ('10s', '15s')")
    ap.add_argument("--months", default="dev")
    ap.add_argument("--tag", default="")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--train_delay", type=int, default=None, help="train on target measured from +D seconds")
    ap.add_argument("--no_extra", action="store_true")
    ap.add_argument("--alts", action="store_true")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--train_days", type=int, default=None)
    ap.add_argument("--features", default=None, help="comma list or @file")
    ap.add_argument("--more", default="", help="extra feature parquet files in data/features (comma list)")
    ap.add_argument("--add", default="", help="prefixes of columns from --more files to add to the feature list")
    a = ap.parse_args()

    df, X = load(extra=not a.no_extra, alts=a.alts, more=a.more)
    if a.features:
        cols = open(a.features[1:]).read().split() if a.features.startswith("@") else a.features.split(",")
        if a.add:
            cols += [c for c in X.columns if c.startswith(tuple(a.add.split(","))) and c not in cols]
        X = X[[c for c in cols if c in X]]
    sub = a.h.endswith("s")
    h = a.h if sub else int(a.h)                       # label used in target column names
    hmin = int(a.h[:-1]) / 60 if sub else h            # horizon in minutes (for the vol scale)
    hrows = 1 if sub else h                            # minutes until the outcome is known (purge)
    scale = F.target_scale(df, hmin)
    if a.venue == "basis":
        bsd = target_series(df, "basis", h).rolling(10080, min_periods=1440).std().shift(h + 1)
        scale = bsd.bfill()
    y_train = target_series(df, a.venue, h, a.train_delay) / scale
    folds = E.make_folds(df["ts"], months_arg(a.months), purge_minutes=hrows + 10, train_days=a.train_days)
    params = json.loads(a.params)
    if a.model == "lgbm":
        fp = M.lgbm_factory(params=params, rounds=a.rounds)
    elif a.model == "ridge":
        fp = M.ridge_factory(alpha=params.get("alpha", 1e4))
    else:
        raise ValueError(a.model)
    t = time.time()
    p = E.walk_forward(fp, X, y_train, folds, verbose=True)
    name = f"{a.model}_{a.venue}{h}{('_' + a.tag) if a.tag else ''}"
    out = pd.DataFrame({"ts": df["ts"], "pred": p, "scale": scale})
    out = out[p.notna()]
    out.to_parquet(PRED / f"{name}_{a.months.replace(',', '-')}.parquet")
    if a.venue in ("s", "f"):
        r = E.summarize(name, df, p, h, venue=a.venue, scale=scale)
    else:
        y = target_series(df, a.venue, h)
        r = {"model": name, **E.metrics(y.values[p.notna()], p.values[p.notna()], scale.values[p.notna()])}
    r["secs"] = round(time.time() - t)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    yv = target_series(df, a.venue, h).values
    print("monthly IC:", E.monthly_ic(df["ts"], yv, p.values).round(3).to_dict(), flush=True)
    if a.model == "lgbm":
        imp = pd.Series(fp.last_model.feature_importance("gain"), index=fp.last_features).sort_values(ascending=False)
        (imp / imp.sum()).to_csv(DEV / f"imp_{name}.csv")
        print("top:", (imp / imp.sum()).head(20).round(3).to_dict(), flush=True)
    with open(DEV / "results.jsonl", "a") as fh:
        fh.write(json.dumps({**r, "months": a.months, "args": vars(a)}) + "\n")


if __name__ == "__main__":
    main()
