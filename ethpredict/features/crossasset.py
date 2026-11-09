"""Cross-asset & cross-book second-level features.

  * crypto "swarm": last-W-second returns of BTC, SOL, XRP, DOGE, BNB (+12 more alts in the
    wide version) and their breadth -- does the crowd move before ETH? (universe in config.py)
  * other ETH spot books on Binance (ETH/USDC): log-price deviation from ETH/USDT and its
    deviation from its own rolling mean (cross-book convergence, like the perp basis)
  * the other major's perpetual (BTC perp): xp_* returns/flow, BTC perp minus ETH perp,
    BTC perp-spot basis deviation
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..config import ALTS, ALTS2, BOOKS, DATA_DIRS, DIR_LEAD_PERP, DIR_PERP, DIR_SPOT, LEAD_KEY
from .seconds import NMIN, PRE, _extended



def day_alt(day: dt.date) -> pd.DataFrame | None:
    S = _extended(DIR_SPOT, day, ["c", "v"])
    if S is None:
        return None
    m = np.arange(NMIN)
    end = PRE + 60 * m + 59
    alts = {a: _extended(DATA_DIRS[a], day, ["c", "v", "bv"]) for a in ALTS}
    books = {b: _extended(DATA_DIRS[b], day, ["c", "v", "n"]) for b in BOOKS}
    return alt_from_arrays(S, alts, books, end, (pd.Timestamp(day) + pd.to_timedelta(60 * m, unit="s")).values)


def alt_from_arrays(S: dict, alts: dict, books: dict, end: np.ndarray, ts: np.ndarray) -> pd.DataFrame:
    """Shared by history and live. Needs >= PRE+60 seconds of history before end[0]."""
    f = {"ts": ts}
    lpS = S["lp"]
    rS = np.diff(lpS, prepend=lpS[0])
    sigS = np.sqrt(pd.Series(rS ** 2).rolling(PRE, min_periods=60).mean().to_numpy() + 1e-14)
    zs = []
    for a in ALTS:
        A = alts.get(a)
        if A is None:
            continue
        lp = A["lp"]
        r = np.diff(lp, prepend=lp[0])
        sig = np.sqrt(pd.Series(r ** 2).rolling(PRE, min_periods=60).mean().to_numpy() + 1e-14)
        for w in (3, 10, 30, 60):
            z = (lp[end] - lp[end - w]) / (sig[end] * np.sqrt(w))
            f[f"alt_{a}_z{w}"] = z
            if w in (10, 30):
                zs.append((w, z))
        cs = np.concatenate([[0.0], np.cumsum(2 * A["bv"] - A["v"])])
        cv = np.concatenate([[0.0], np.cumsum(A["v"])])
        f[f"alt_{a}_imb30"] = (cs[end + 1] - cs[end + 1 - 30]) / (cv[end + 1] - cv[end + 1 - 30] + 1e-12)
    for w in (10, 30):
        arr = np.stack([z for ww, z in zs if ww == w])
        f[f"alt_swarm_z{w}"] = np.nanmean(arr, 0)
        f[f"alt_swarm_breadth{w}"] = np.nanmean(np.sign(arr), 0)
        self_z = (lpS[end] - lpS[end - w]) / (sigS[end] * np.sqrt(w))   # the target asset itself
        f[f"alt_swarm_minus_self{w}"] = f[f"alt_swarm_z{w}"] - self_z
    for b in BOOKS:
        B = books.get(b)
        if B is None:
            continue
        dev = B["lp"] - lpS
        cs = np.concatenate([[0.0], np.cumsum(dev)])
        f[f"book_{b}_dev"] = dev[end] / (sigS[end] * np.sqrt(60))
        for w in (10, 60, 300):
            f[f"book_{b}_dev{w}"] = (dev[end] - (cs[end + 1] - cs[end + 1 - w]) / w) / (sigS[end] * np.sqrt(60))
        rb = B["lp"][end] - B["lp"][end - 10]
        f[f"book_{b}_r10"] = rb / (sigS[end] * np.sqrt(10))
        cn = np.concatenate([[0.0], np.cumsum(B["n"])])
        f[f"book_{b}_ln60"] = np.log1p(cn[end + 1] - cn[end + 1 - 60])
    return pd.DataFrame(f)


def build(start: str, end: str, workers: int = 4) -> pd.DataFrame:
    from multiprocessing import Pool

    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    days = [s + dt.timedelta(days=i) for i in range((e - s).days + 1)]
    with Pool(workers) as pool:
        parts = [p for p in pool.imap(day_alt, days, chunksize=4) if p is not None]
    df = pd.concat(parts, ignore_index=True)
    num = df.columns.drop("ts")
    df[num] = df[num].astype("float32").replace([np.inf, -np.inf], np.nan)
    return df


# ---------------------------------------------------------------------------------------------
# v4: wider altcoin swarm + ETH perpetual


def alt2_from_arrays(S: dict, Fb: dict | None, alts: dict, xperp: dict | None, end: np.ndarray, ts: np.ndarray) -> pd.DataFrame:
    """S: target spot, Fb: target perp, alts: all swarm spot arrays (ALTS + ALTS2),
    xperp: the other major's perpetual (BTC perp for ETH) -> xp_* features."""
    f = {"ts": ts}
    Z = {w: [] for w in (3, 10, 30)}
    for a, A in alts.items():
        if A is None or not np.isfinite(A["lp"]).any():
            continue
        lp = A["lp"]
        r = np.diff(lp, prepend=lp[0])
        sig = np.sqrt(pd.Series(r ** 2).rolling(PRE, min_periods=60).mean().to_numpy() + 1e-14)
        for w in (3, 10, 30):
            z = (lp[end] - lp[end - w]) / (sig[end] * np.sqrt(w))
            # coins that did not trade in the window carry no information
            Z[w].append(np.where(np.abs(lp[end] - lp[end - w]) > 0, z, np.nan))
    for w in (3, 10, 30):
        arr = np.stack(Z[w])
        f[f"swarm2_breadth{w}"] = np.nanmean(np.sign(arr), 0)
        f[f"swarm2_mean{w}"] = np.nanmean(np.clip(arr, -5, 5), 0)
        f[f"swarm2_med{w}"] = np.nanmedian(arr, 0)
        f[f"swarm2_active{w}"] = np.mean(np.isfinite(arr), 0)
    if xperp is not None:
        lp = xperp["lp"]
        r = np.diff(lp, prepend=lp[0])
        sig = np.sqrt(pd.Series(r ** 2).rolling(PRE, min_periods=60).mean().to_numpy() + 1e-14)
        flow = np.concatenate([[0.0], np.cumsum(2 * xperp["bv"] - xperp["v"])])
        vol = np.concatenate([[0.0], np.cumsum(xperp["v"])])
        for w in (1, 3, 10, 30, 60):
            f[f"xp_z{w}"] = (lp[end] - lp[end - w]) / (sig[end] * np.sqrt(w))
        for w in (3, 10, 30):
            f[f"xp_imb{w}"] = (flow[end + 1] - flow[end + 1 - w]) / (vol[end + 1] - vol[end + 1 - w] + 1e-12)
        if Fb is not None:
            lpb = Fb["lp"]
            rb = np.diff(lpb, prepend=lpb[0])
            sigb = np.sqrt(pd.Series(rb ** 2).rolling(PRE, min_periods=60).mean().to_numpy() + 1e-14)
            for w in (3, 10, 30):
                f[f"xp_minus_tp{w}"] = f[f"xp_z{w}"] - (lpb[end] - lpb[end - w]) / (sigb[end] * np.sqrt(w))
        es = alts.get(LEAD_KEY)
        if es is not None:
            eb = lp - es["lp"]
            cs = np.concatenate([[0.0], np.cumsum(eb)])
            for w in (10, 60):
                f[f"x_basis_dev{w}"] = (eb[end] - (cs[end + 1] - cs[end + 1 - w]) / w) / (sig[end] * np.sqrt(60))
    return pd.DataFrame(f)


def day_alt2(day: dt.date) -> pd.DataFrame | None:
    S = _extended(DIR_SPOT, day, ["c", "v"])
    if S is None:
        return None
    Fb = _extended(DIR_PERP, day, ["c", "v"])
    m = np.arange(NMIN)
    end = PRE + 60 * m + 59
    alts = {a: _extended(DATA_DIRS[a], day, ["c", "v", "bv"]) for a in ALTS + ALTS2}
    xperp = _extended(DIR_LEAD_PERP, day, ["c", "v", "bv"])
    return alt2_from_arrays(S, Fb, alts, xperp, end, (pd.Timestamp(day) + pd.to_timedelta(60 * m, unit="s")).values)


def build2(start: str, end: str, workers: int = 3) -> pd.DataFrame:
    from multiprocessing import Pool

    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    days = [s + dt.timedelta(days=i) for i in range((e - s).days + 1)]
    with Pool(workers) as pool:
        parts = [p for p in pool.imap(day_alt2, days, chunksize=4) if p is not None]
    df = pd.concat(parts, ignore_index=True)
    num = df.columns.drop("ts")
    df[num] = df[num].astype("float32").replace([np.inf, -np.inf], np.nan)
    return df
