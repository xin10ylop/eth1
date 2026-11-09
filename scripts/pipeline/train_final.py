"""Final pipeline: honest HOLDOUT walk-forward evaluation + production models + calibration.

Architecture per target (venue, horizon):
  CORE      LightGBM on ETH spot/perp + BTC microstructure features (robust)
  SATELLITE LightGBM on core + cross-venue/alt-coin features (stronger, more fragile)
  ENSEMBLE  online discounted ridge (half-life 3 days) re-weighting [1, core, satellite] every
            minute from realized outcomes, shrunk toward "trust the satellite" by a persistent
            prior -> falls back to the core within days if a cross-venue relationship breaks

For every variant/target:
  1. walk-forward over the 6 HOLDOUT months (never used for any design decision), monthly retrain
  2. RLS ensemble on those OOS forecasts (causal); calibration artifacts from the ensemble
  3. production core+satellite trained on ALL data; RLS state saved for live continuation
Outputs: models/*.txt, models/meta.json, reports/holdout/holdout_*.json, reports/preds/final_*.parquet
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
from ethpredict.modeling import calibrate as C  # noqa: E402
from ethpredict.modeling import evaluate as E  # noqa: E402
from ethpredict.features import matrix as F  # noqa: E402
from ethpredict.modeling import gbm as M  # noqa: E402
from ethpredict.modeling import online  # noqa: E402
from ethpredict.config import CONFIRM, FEAT, FEATURE_SETS, HOLDOUT, MODELS, PREDS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="full,spot")
ap.add_argument("--horizons", default="10s,15s,1,2,3,4,5")
ap.add_argument("--venues", default="f,s")
ap.add_argument("--rounds", type=int, default=400)
ap.add_argument("--params", default="{}")
ap.add_argument("--months", default="holdout")
ap.add_argument("--half_life", type=float, default=4320)
ap.add_argument("--kappa", type=float, default=10.0, help="strength of the persistent prior 'trust the satellite'")
ap.add_argument("--block", type=int, default=2, help="test months per retrain")
ap.add_argument("--skip_final", action="store_true")
ap.add_argument("--production_only", action="store_true",
                help="monthly refresh: retrain core+satellite on ALL data, keep the validated calibration/RLS state")
ap.add_argument("--fast_from", type=int, default=2,
                help="horizons (minutes) >= this use the *_fast feature sets (sub-minute and 1 m always use the full sets)")
ap.add_argument("--extra_x", default="Xcb.parquet,Xalt2.parquet", help="comma list of extra feature parquet files in data/features")
a = ap.parse_args()

VARIANTS = {
    "full": {"core": str(FEATURE_SETS / "core.txt"), "sat": str(FEATURE_SETS / "sat.txt"), "uses_futures": True, "venues": ["f", "s"]},
    "spot": {"core": str(FEATURE_SETS / "spotcore.txt"), "sat": str(FEATURE_SETS / "spotsat.txt"), "uses_futures": False, "venues": ["s"]},
}

df = pd.read_parquet(FEAT / "panel.parquet")
df = df[df["ts"] >= "2025-05-03"].reset_index(drop=True)
if (FEAT / "Ysub.parquet").exists():          # sub-minute targets (10s, 15s)
    df = pd.concat([df, pd.read_parquet(FEAT / "Ysub.parquet")], axis=1)
parts = [pd.read_parquet(FEAT / "X.parquet"), F.extra_features(df), pd.read_parquet(FEAT / "Xalt.parquet")]
for f in [x for x in a.extra_x.split(",") if x]:
    parts.append(pd.read_parquet(FEAT / f))
X = pd.concat(parts, axis=1)
months = E.HOLDOUT_MONTHS if a.months == "holdout" else a.months.split(",")
PRED = PREDS
meta_path = MODELS / "meta.json" if a.months == "holdout" else CONFIRM / f"meta_{a.months.replace(',', '-')}.json"
meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
params = json.loads(a.params)
mtag = a.months.replace(",", "-")


def calib_artifacts(z, y, scale):
    from sklearn.linear_model import LogisticRegression
    ok = np.isfinite(y) & (y != 0) & np.isfinite(z)
    lr = LogisticRegression(C=1e3).fit(z[ok].reshape(-1, 1), (y[ok] > 0).astype(int))
    yzs = y / scale
    fin = np.isfinite(yzs) & np.isfinite(z)
    beta = float(np.sum(z[fin] * yzs[fin]) / np.sum(z[fin] ** 2))
    resid = yzs[fin] - beta * z[fin]
    absz = np.abs(z[np.isfinite(z)])
    dec = pd.qcut(np.abs(z[ok]), 10, labels=False, duplicates="drop")
    hit_by = pd.Series(np.sign(z[ok]) == np.sign(y[ok])).groupby(dec).mean().round(4).tolist()
    return {"calib": [float(lr.intercept_[0]), float(lr.coef_[0, 0])], "beta": beta,
            "resid_q": {str(q): float(np.quantile(resid, q)) for q in (0.025, 0.1, 0.5, 0.9, 0.975)},
            "absz_quantiles": np.quantile(absz, np.linspace(0.01, 0.99, 99)).round(5).tolist(),
            "hit_by_conf_decile": hit_by}


for variant in a.variants.split(","):
    spec = VARIANTS[variant]
    cols = {}
    for part in ("core", "sat"):
        cols[part] = [c for c in open(spec[part]).read().split() if c in X.columns]
        cols[part + "_fast"] = [c for c in open(spec[part].replace(".txt", "_fast.txt")).read().split() if c in X.columns]
    vm = meta.setdefault(variant, {"targets": {}})
    vm.update({"uses_futures": spec["uses_futures"], "half_life": a.half_life})
    for venue in [v for v in spec["venues"] if v in a.venues.split(",")]:
        for tok in a.horizons.split(","):
            # minute horizons "1".."5" or second horizons "10s", "15s"
            sub = tok.endswith("s")
            hmin = int(tok[:-1]) / 60 if sub else int(tok)        # horizon in minutes
            hlab = tok
            h = 1 if sub else int(tok)                            # rows until the outcome is known / feature-set switch
            key = f"{venue}{hlab}"
            ycol = f"y_{venue}{hlab}"
            t0 = time.time()
            scale = F.target_scale(df, hmin)
            yz = df[ycol] / scale
            if a.production_only:
                entry = vm["targets"][key]
                hc = {part: (cols[part] if h < a.fast_from else cols[part + "_fast"]) for part in ("core", "sat")}
                tr = np.where(np.isfinite(yz.values))[0]
                for part in ("core", "sat"):
                    fp = M.lgbm_factory(params=params, rounds=a.rounds)
                    fp(X[hc[part]].iloc[tr], yz.iloc[tr], X[hc[part]].iloc[-5:], None)
                    fp.last_model.save_model(str(MODELS / entry["files"][part]))
                    entry.setdefault("features", {})[part] = hc[part]
                entry["trained_through"] = str(df["ts"].iloc[tr[-1]])
                meta_path.write_text(json.dumps(meta, indent=1, default=float))
                print(f"{variant} {key} retrained through {entry['trained_through']} ({time.time()-t0:.0f}s)", flush=True)
                continue
            folds = E.make_folds(df["ts"], months, purge_minutes=h + 10, block=a.block)
            preds, fps = {}, {}
            hc = {part: (cols[part] if h < a.fast_from else cols[part + "_fast"]) for part in ("core", "sat")}
            for part in ("core", "sat"):
                fps[part] = M.lgbm_factory(params=params, rounds=a.rounds)
                preds[part] = E.walk_forward(fps[part], X[hc[part]], yz, folds, verbose=False)
            m = preds["core"].notna().values & preds["sat"].notna().values
            # inputs [core, sat] without intercept (a drifting intercept is a 3-day trend forecast: noise).
            # h=1: adaptive (nearly independent outcomes, fragile seconds-scale lead-lag);
            # h>=2: frozen at the satellite (overlapping outcomes make online weights too noisy).
            Z = np.column_stack([preds["core"].values[m], preds["sat"].values[m]])
            kappa = a.kappa if h == 1 else 1e12
            ens, rls = online.prior_stack(Z, yz.values[m], horizon=h, half_life=a.half_life, kappa=kappa,
                                          prior=np.r_[0.0, 1.0], return_model=True)
            ens_s = pd.Series(np.nan, index=df.index); ens_s[m] = ens
            out = pd.DataFrame({"ts": df["ts"][m], "core": preds["core"][m], "sat": preds["sat"][m], "pred": ens, "scale": scale[m]})
            out.to_parquet(PRED / f"final_{variant}_{key}_{mtag}.parquet")
            res = {}
            for name, p in (("core", preds["core"]), ("sat", preds["sat"]), ("ens", ens_s)):
                pp = p.where(pd.Series(m, index=df.index))
                r = E.summarize(f"{variant}_{key}_{name}", df, pp, hlab, venue=venue, scale=scale)
                r["monthly_ic"] = E.monthly_ic(df["ts"], df[ycol].values, pp.values).round(4).to_dict()
                res[name] = r
            y = df[ycol].values[m]
            pu = C.causal_logistic(ens, y, horizon=h)
            res["ens_calibration"] = C.logloss(pu, y)
            res["ens_vs_core"] = E.paired_block_bootstrap(df["ts"][m], y, preds["core"].values[m], ens, n_boot=500)
            res["ens_vs_sat"] = E.paired_block_bootstrap(df["ts"][m], y, preds["sat"].values[m], ens, n_boot=500)
            entry = {"venue": venue, "h": hmin, "label": hlab, "files": {"core": f"{variant}_{key}_core.txt", "sat": f"{variant}_{key}_sat.txt"},
                     **calib_artifacts(ens, y, scale.values[m]), "rls_state": rls.state(),
                     "holdout": {n: {k: v for k, v in r.items()} for n, r in res.items() if n in ("core", "sat", "ens")},
                     "holdout_logloss_skill_pct": res["ens_calibration"]["logloss_skill_pct"]}
            if not a.skip_final:
                tr = np.where(np.isfinite(yz.values))[0]
                for part in ("core", "sat"):
                    fps[part](X[hc[part]].iloc[tr], yz.iloc[tr], X[hc[part]].iloc[-5:], None)
                    fps[part].last_model.save_model(str(MODELS / entry["files"][part]))
            entry["features"] = {part: hc[part] for part in ("core", "sat")}
            vm["targets"][key] = entry
            meta_path.write_text(json.dumps(meta, indent=1, default=float))
            ((HOLDOUT if a.months == "holdout" else CONFIRM) / f"{'holdout' if a.months == 'holdout' else 'confirm'}_{variant}_{key}.json").write_text(json.dumps(res, indent=1, default=float))
            for name in ("core", "sat", "ens"):
                r = res[name]
                print(f"{variant} {key} {name:4s} ic={r['ic']:.4f} hit={r['hit']:.4f} auc={r['auc']:.4f} top10={r['hit_top10']:.4f} "
                      f"top1={r['hit_top1']:.4f} ic_d2={r['ic_d2']:.4f} ic_d5={r['ic_d5']:.4f} monthly={r['monthly_ic']}", flush=True)
            print(f"   ens-core dIC={res['ens_vs_core']['diff']:+.4f} p={res['ens_vs_core']['p_le_0']:.3f} | ens-sat dIC={res['ens_vs_sat']['diff']:+.4f} "
                  f"p={res['ens_vs_sat']['p_le_0']:.3f} | logloss skill {res['ens_calibration']['logloss_skill_pct']:.2f}% | {time.time()-t0:.0f}s", flush=True)
