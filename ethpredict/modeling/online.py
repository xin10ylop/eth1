"""Online adaptive layer: recursive least squares with exponential forgetting.

Combines a robust CORE model (target spot/perp + the other major's spot only) and an opportunistic SATELLITE model
(cross-venue / alt-coin features) minute by minute, using only targets that have already been
realized. When a cross-venue relationship breaks (e.g. BTC/FDUSD liquidity vanished in Feb 2026)
the weight flows back to the core within days instead of waiting for the next retrain.
"""
from __future__ import annotations

import numpy as np


class OnlineRLS:
    def __init__(self, k: int, half_life: float = 2880.0, ridge: float = 1e4, prior=None):
        self.lam = 0.5 ** (1.0 / half_life)
        self.ridge = ridge
        self.P = np.eye(k) / ridge
        self.w = np.zeros(k) if prior is None else np.asarray(prior, float).copy()
        self.k = k
        self.n = 0

    def predict(self, x) -> float:
        return float(np.nan_to_num(np.asarray(x, float)) @ self.w)

    def update(self, x, y: float, clip: float = 6.0):
        if not np.isfinite(y):
            return
        x = np.nan_to_num(np.asarray(x, float))
        y = float(np.clip(y, -clip, clip))
        Px = self.P @ x
        g = Px / (self.lam + x @ Px)
        self.w = self.w + g * (y - x @ self.w)
        self.P = (self.P - np.outer(g, Px)) / self.lam
        self.n += 1
        if self.n % 1000 == 0:          # numerical hygiene
            self.P = 0.5 * (self.P + self.P.T)
            tr = np.trace(self.P)
            cap = 1e3 * self.k / self.ridge
            if tr > cap:
                self.P *= cap / tr

    def state(self) -> dict:
        return {"w": self.w.tolist(), "P": self.P.tolist(), "lam": self.lam, "ridge": self.ridge, "n": self.n}

    @classmethod
    def from_state(cls, s: dict) -> "OnlineRLS":
        o = cls(len(s["w"]))
        o.w = np.asarray(s["w"], float); o.P = np.asarray(s["P"], float)
        o.lam, o.ridge, o.n = s["lam"], s["ridge"], s["n"]
        return o


def rls_stack(Z: np.ndarray, y: np.ndarray, horizon: int, half_life: float = 2880.0, ridge: float = 1e4,
              prior: np.ndarray | None = None, valid: np.ndarray | None = None, return_model: bool = False):
    """Z: (N, k) inputs ordered in time (one row per minute); y: (N,) realized z-target.
    Row t uses only y[s] for s <= t - horizon (those targets are realized at the bar close of t)."""
    n, k = Z.shape
    m = OnlineRLS(k, half_life, ridge, prior)
    out = np.full(n, np.nan)
    valid = np.ones(n, bool) if valid is None else valid
    for t in range(n):
        out[t] = m.predict(Z[t])
        s = t - horizon
        if s >= 0 and valid[s]:
            m.update(Z[s], y[s])
    return (out, m) if return_model else out


class PriorRLS:
    """Discounted ridge with a PERSISTENT prior:
        w_t = argmin  sum_s lam^(t-s) (y_s - x_s.w)^2  +  kappa * ||w - w0||^2
    Unlike plain RLS (whose initial prior is forgotten), the pull toward w0 never decays, so the
    weights only leave the prior when recent evidence clearly contradicts it (e.g. a venue break)."""

    def __init__(self, k: int, half_life: float = 10080.0, kappa: float = 3000.0, prior=None):
        self.lam = 0.5 ** (1.0 / half_life)
        self.kappa = kappa
        self.w0 = np.zeros(k) if prior is None else np.asarray(prior, float).copy()
        self.A = np.zeros((k, k))
        self.b = np.zeros(k)
        self.k = k
        self.n = 0
        self.w = self.w0.copy()

    def predict(self, x) -> float:
        return float(np.nan_to_num(np.asarray(x, float)) @ self.w)

    def update(self, x, y: float, clip: float = 6.0, refresh: int = 30):
        if not np.isfinite(y):
            return
        x = np.nan_to_num(np.asarray(x, float))
        y = float(np.clip(y, -clip, clip))
        self.A = self.lam * self.A + np.outer(x, x)
        self.b = self.lam * self.b + x * y
        self.n += 1
        if self.n % refresh == 0:
            self.w = np.linalg.solve(self.A + self.kappa * np.eye(self.k), self.b + self.kappa * self.w0)

    def state(self) -> dict:
        return {"type": "prior", "w": self.w.tolist(), "A": self.A.tolist(), "b": self.b.tolist(), "lam": self.lam,
                "kappa": self.kappa, "w0": self.w0.tolist(), "n": self.n}

    @classmethod
    def from_state(cls, s: dict) -> "PriorRLS":
        o = cls(len(s["w"]))
        o.w = np.asarray(s["w"], float); o.A = np.asarray(s["A"], float); o.b = np.asarray(s["b"], float)
        o.lam, o.kappa, o.w0, o.n = s["lam"], s["kappa"], np.asarray(s["w0"], float), s["n"]
        return o


def prior_stack(Z, y, horizon, half_life=10080.0, kappa=3000.0, prior=None, return_model=False):
    n, k = Z.shape
    m = PriorRLS(k, half_life, kappa, prior)
    out = np.full(n, np.nan)
    for t in range(n):
        out[t] = m.predict(Z[t])
        s = t - horizon
        if s >= 0:
            m.update(Z[s], y[s])
    return (out, m) if return_model else out
