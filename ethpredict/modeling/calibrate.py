"""Probability calibration and distributional forecasts.

* P(up): causal logistic calibration of the z-score forecast (refit as data accrues)
* Intervals: adaptive conformal inference (ACI, Gibbs & Candes 2021) on top of a volatility
  forecast -> intervals keep their coverage under regime change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def causal_logistic(pred: np.ndarray, y: np.ndarray, horizon: int, refit_every: int = 1440,
                    min_obs: int = 20000, window: int = 200000):
    """Return P(y>0) using a 2-parameter logistic map a + b*pred refit every `refit_every` rows on
    the most recent `window` rows whose target is already known."""
    from sklearn.linear_model import LogisticRegression

    n = len(pred)
    out = np.full(n, np.nan)
    a, b = 0.0, None
    lab = (y > 0).astype(float)
    ok_all = np.isfinite(pred) & np.isfinite(y) & (y != 0)
    for start in range(0, n, refit_every):
        known = start - horizon
        if known > min_obs:
            lo = max(0, known - window)
            m = ok_all[lo:known]
            if m.sum() > min_obs:
                lr = LogisticRegression(C=1e3)
                lr.fit(pred[lo:known][m].reshape(-1, 1), lab[lo:known][m])
                a, b = lr.intercept_[0], lr.coef_[0, 0]
        if b is not None:
            z = a + b * pred[start:start + refit_every]
            out[start:start + refit_every] = 1 / (1 + np.exp(-z))
    return out


def logloss(p: np.ndarray, y: np.ndarray) -> dict:
    m = np.isfinite(p) & np.isfinite(y) & (y != 0)
    p, lab = np.clip(p[m], 1e-6, 1 - 1e-6), (y[m] > 0).astype(float)
    ll = -np.mean(lab * np.log(p) + (1 - lab) * np.log(1 - p))
    base = lab.mean()
    ll0 = -np.mean(lab * np.log(base) + (1 - lab) * np.log(1 - base))
    brier = np.mean((p - lab) ** 2)
    # reliability: mean predicted vs realized in deciles
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    rel = pd.DataFrame({"q": q, "p": p, "y": lab}).groupby("q").mean()
    return {"logloss": ll, "logloss_base": ll0, "logloss_skill_pct": 100 * (ll0 - ll) / ll0, "brier": brier,
            "reliability": rel.round(4).values.tolist()}


def aci_intervals(center: np.ndarray, scale: np.ndarray, y: np.ndarray, horizon: int, alpha: float = 0.1,
                  gamma: float = 0.005, warm: int = 5000):
    """Adaptive conformal intervals around `center` with width q_t * scale_t.
    Nonconformity score s = |y - center| / scale. alpha_t updated online:
        alpha_{t+1} = alpha_t + gamma * (alpha - err_t)   (err_t uses targets known at t)
    Returns lower, upper, and realized coverage indicator."""
    n = len(y)
    s = np.abs(y - center) / scale
    lo = np.full(n, np.nan); hi = np.full(n, np.nan)
    a_t = alpha
    hist: list[float] = []
    covered = np.full(n, np.nan)
    from collections import deque
    buf = deque(maxlen=20000)
    for t in range(n):
        k = t - horizon
        if k >= 0 and np.isfinite(s[k]):
            buf.append(s[k])
            if np.isfinite(hi[k]):
                err = float(not (lo[k] <= y[k] <= hi[k]))
                a_t = a_t + gamma * (alpha - err)
                a_t = min(max(a_t, 0.001), 0.5)
        if len(buf) >= warm:
            if t % 60 == 0 or not hist:
                arr = np.fromiter(buf, float)
                hist = [np.quantile(arr, 1 - a_t)]
            q = hist[0]
            lo[t] = center[t] - q * scale[t]
            hi[t] = center[t] + q * scale[t]
            if np.isfinite(y[t]):
                covered[t] = float(lo[t] <= y[t] <= hi[t])
    return lo, hi, covered
