"""Walk-forward evaluation with purging.

Protocol
  * expanding training window, retrain once per test month
  * training rows whose target window overlaps the test period are purged (+ embargo)
  * DEV months are used for every design decision; HOLDOUT months are touched only for the
    final report.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

DEV_MONTHS = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"]
HOLDOUT_MONTHS = ["2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]


@dataclass
class Fold:
    name: str
    train: np.ndarray
    test: np.ndarray


def make_folds(ts: pd.Series, months: list[str], purge_minutes: int = 15, min_train_days: int = 60,
               train_days: int | None = None, valid: np.ndarray | None = None, block: int = 1) -> list[Fold]:
    """One fold per `block` consecutive test months; training ends `purge_minutes` before the block."""
    ts = pd.DatetimeIndex(ts)
    valid = np.ones(len(ts), bool) if valid is None else valid
    out = []
    groups = [months[i:i + block] for i in range(0, len(months), block)]
    for g in groups:
        m = g[0]
        start = pd.Timestamp(g[0] + "-01")
        end = pd.Timestamp(g[-1] + "-01") + pd.offsets.MonthBegin(1)
        te = np.where((ts >= start) & (ts < end) & valid)[0]
        cut = start - pd.Timedelta(minutes=purge_minutes)
        lo = ts[0] if train_days is None else cut - pd.Timedelta(days=train_days)
        tr = np.where((ts < cut) & (ts >= lo) & valid)[0]
        if len(te) == 0 or len(tr) == 0 or (ts[tr].max() - ts[tr].min()).days < min_train_days:
            continue
        out.append(Fold(m, tr, te))
    return out


def metrics(y: np.ndarray, p: np.ndarray, scale: np.ndarray | None = None) -> dict:
    """y: realized log return; p: prediction (any monotone score; z-units if scale is given)."""
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    res = {"n": int(len(y))}
    res["ic"] = float(np.corrcoef(p, y)[0, 1])
    res["rank_ic"] = float(spearmanr(p, y).statistic)
    nz = y != 0
    res["hit"] = float(np.mean(np.sign(p[nz]) == np.sign(y[nz])))
    res["auc"] = float(roc_auc_score(y[nz] > 0, p[nz]))
    if scale is not None:
        s = scale[m]
        pr = p * s
        res["r2_oos"] = float(1 - np.sum((y - pr) ** 2) / np.sum(y ** 2))
    a = np.abs(p)
    for q in (0.5, 0.8, 0.9, 0.95, 0.99):
        sel = (a >= np.quantile(a, q)) & nz
        res[f"hit_top{int(round((1-q)*100))}"] = float(np.mean(np.sign(p[sel]) == np.sign(y[sel])))
    # average absolute move captured when following the sign (bps), all and top decile
    g = np.sign(p) * y * 1e4
    res["edge_bps"] = float(g.mean())
    sel = a >= np.quantile(a, 0.9)
    res["edge_bps_top10"] = float(g[sel].mean())
    return res


def monthly_ic(ts: pd.Series, y: np.ndarray, p: np.ndarray) -> pd.Series:
    d = pd.DataFrame({"m": pd.DatetimeIndex(ts).strftime("%Y-%m"), "y": y, "p": p}).dropna()
    return d.groupby("m").apply(lambda g: np.corrcoef(g.p, g.y)[0, 1], include_groups=False)


def walk_forward(fit_predict, X: pd.DataFrame, yz: pd.Series, folds: list[Fold], verbose=True,
                 sample_weight: pd.Series | None = None) -> pd.Series:
    """fit_predict(Xtr, ytr, Xte, wtr) -> np.ndarray of predictions for Xte."""
    pred = pd.Series(np.nan, index=X.index)
    for f in folds:
        t = time.time()
        tr = f.train[np.isfinite(yz.values[f.train])]
        w = None if sample_weight is None else sample_weight.values[tr]
        pred.iloc[f.test] = fit_predict(X.iloc[tr], yz.iloc[tr], X.iloc[f.test], w)
        if verbose:
            print(f"  fold {f.name}: train={len(tr)} test={len(f.test)} {time.time()-t:.0f}s", flush=True)
    return pred


def summarize(name: str, df: pd.DataFrame, pred: pd.Series, h: int, venue="s", delays=(None, 1, 2, 5),
              scale: pd.Series | None = None) -> dict:
    """Metrics of OOS predictions against the target measured with several execution delays."""
    m = pred.notna().values
    out = {"model": name, "h": h}
    for d in delays:
        col = f"y_{venue}{h}" if d is None else f"y_{venue}{h}_d{d}"
        y = df[col].values[m]
        r = metrics(y, pred.values[m], None if scale is None else scale.values[m])
        tag = "" if d is None else f"_d{d}"
        if d is None:
            out.update(r)
        else:
            out[f"ic{tag}"] = r["ic"]
            out[f"hit{tag}"] = r["hit"]
    return out


def paired_block_bootstrap(ts: pd.Series, y: np.ndarray, p1: np.ndarray, p2: np.ndarray, n_boot: int = 2000,
                           block: str = "D", seed: int = 0) -> dict:
    """IC(p2) - IC(p1) with a day-block bootstrap (keeps intraday dependence intact).
    Returns the point difference, a 95% CI and the one-sided p-value of (diff <= 0)."""
    d = pd.DataFrame({"g": pd.DatetimeIndex(ts).floor(block), "y": y, "a": p1, "b": p2}).dropna()
    # per-block sufficient statistics for correlations
    def stats(x):
        return pd.Series({"n": len(x), "sy": x.y.sum(), "syy": (x.y ** 2).sum(),
                          "sa": x.a.sum(), "saa": (x.a ** 2).sum(), "say": (x.a * x.y).sum(),
                          "sb": x.b.sum(), "sbb": (x.b ** 2).sum(), "sby": (x.b * x.y).sum()})
    S = d.groupby("g").apply(stats, include_groups=False).values
    def corr_diff(T):
        n, sy, syy, sa, saa, say, sb, sbb, sby = T
        vy = syy / n - (sy / n) ** 2
        ca = (say / n - sa * sy / n ** 2) / np.sqrt((saa / n - (sa / n) ** 2) * vy)
        cb = (sby / n - sb * sy / n ** 2) / np.sqrt((sbb / n - (sb / n) ** 2) * vy)
        return cb - ca
    point = corr_diff(S.sum(0))
    rng = np.random.default_rng(seed)
    k = len(S)
    boots = np.array([corr_diff(S[rng.integers(0, k, k)].sum(0)) for _ in range(n_boot)])
    return {"diff": float(point), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "p_le_0": float(np.mean(boots <= 0)), "blocks": int(k)}
