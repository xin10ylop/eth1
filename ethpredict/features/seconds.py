"""Second-level (intra-minute) feature extraction.

For every one-minute bar m of a day we build features that use ONLY seconds strictly before the
prediction instant e = open_time(m) + 60s, plus forward-looking target prices that are stored in
clearly separated `y_*` / `px_*` columns.

Inputs are the per-second bars written by `data/binance.py` (roles set in config.py):
    S  target spot     ETHUSDT 1s klines           -> s_* features      o h l c v qv n bv
    F  target perp     ETHUSDT aggTrades           -> f_* features      + nb lbv lsv mxq
    E  external major  BTCUSDT spot 1s klines      -> e_* features      (BTC leads ETH)

Key ideas implemented here
  * multi-window returns / signed order flow on both venues (1s ... 300s)
  * perp-vs-spot micro-basis and its deviation from its own recent mean (lead-lag catch-up)
  * Levy areas (2nd-order path-signature terms) between pairs of paths:
        futures-vs-spot price, price-vs-cumulative-flow, target-vs-external major
    The signed area tells *who led whom* inside the window, which plain returns cannot see.
  * "first-touch" shape stats inside the last minute (where in the minute the move happened)
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..config import DIR_LEAD_SPOT, DIR_PERP, DIR_SPOT, SEC

PRE = 900     # seconds of history carried in from previous day
POST = 600    # seconds of the next day (targets only)
DAY = 86400
NMIN = 1440

WINDOWS = (1, 2, 3, 5, 10, 20, 30, 60, 120, 300, 600)
LATENCY = (1, 2, 5, 10, 30)


def _load(venue: str, day: dt.date) -> pd.DataFrame | None:
    p = SEC / venue / f"{day}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _extended(venue: str, day: dt.date, cols) -> dict[str, np.ndarray] | None:
    cur = _load(venue, day)
    if cur is None:
        return None
    prev = _load(venue, day - dt.timedelta(days=1))
    nxt = _load(venue, day + dt.timedelta(days=1))
    parts = []
    if prev is not None:
        parts.append(prev.iloc[-PRE:])
    else:
        parts.append(pd.DataFrame({c: np.full(PRE, np.nan if c in "ohlc" else 0.0) for c in cur.columns}))
    parts.append(cur)
    if nxt is not None:
        parts.append(nxt.iloc[:POST])
    else:
        parts.append(pd.DataFrame({c: np.full(POST, np.nan if c in "ohlc" else 0.0) for c in cur.columns}))
    df = pd.concat(parts, ignore_index=True)
    out = {c: df[c].to_numpy(dtype=np.float64) for c in cols if c in df}
    c = pd.Series(df["c"].to_numpy()).ffill().bfill().to_numpy()
    out["lp"] = np.log(c)
    out["has_next"] = nxt is not None
    return out


def _wsum(cs: np.ndarray, end: np.ndarray, w: int) -> np.ndarray:
    """sum of x over indices [end-w+1, end] given cs = cumsum(x) with cs[-1] convention (prepend 0)."""
    return cs[end + 1] - cs[end + 1 - w]


def _cs(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(x)])


def _levy(X: np.ndarray, Y: np.ndarray, end: np.ndarray, w: int):
    """Levy area of the 2-d path (X,Y) over points [end-w, end] (w increments).

    A = 1/2 * sum_i [(X_{i-1}-X_a) dY_i - (Y_{i-1}-Y_a) dX_i]
    Computed with prefix sums -> O(1) per window.
    """
    dX = np.diff(X, prepend=X[0])
    dY = np.diff(Y, prepend=Y[0])
    Xl = np.concatenate([[X[0]], X[:-1]])
    Yl = np.concatenate([[Y[0]], Y[:-1]])
    c1 = _cs(Xl * dY)
    c2 = _cs(Yl * dX)
    a = end - w
    s1 = c1[end + 1] - c1[a + 1] - X[a] * (Y[end] - Y[a])
    s2 = c2[end + 1] - c2[a + 1] - Y[a] * (X[end] - X[a])
    return 0.5 * (s1 - s2)


def day_features(day: dt.date) -> pd.DataFrame | None:
    S = _extended(DIR_SPOT, day, ["o", "h", "l", "c", "v", "n", "bv", "qv"])
    F = _extended(DIR_PERP, day, ["o", "h", "l", "c", "v", "n", "bv", "lbv", "lsv", "mxq", "qv"])
    if S is None or F is None:
        return None
    E = _extended(DIR_LEAD_SPOT, day, ["c", "v", "bv"])

    t0 = pd.Timestamp(day, tz="UTC")
    m = np.arange(NMIN)
    end = PRE + 60 * m + 59              # last second index belonging to bar m (info boundary)
    out = features_from_arrays(S, F, E, end, (t0 + pd.to_timedelta(60 * m, unit="s")).values, forward=True)
    if not S["has_next"]:
        # last day available: forward latency prices for the last minute are unknown
        for d in LATENCY:
            out.loc[out.index[-1], [f"px_s_d{d}", f"px_f_d{d}"]] = np.nan
    return out


def features_from_arrays(S: dict, F: dict, E: dict | None, end: np.ndarray, ts: np.ndarray,
                         forward: bool = False) -> pd.DataFrame:
    """Core feature computation on per-second arrays (dicts with o h l c v n bv ... and 'lp').
    `end` = index of the last second of each bar. Needs >= 660 seconds of history before end[0].
    Used both for history (day files) and live buffers, so training and live share one code path."""
    f: dict[str, np.ndarray] = {"ts": ts}

    # center log prices for numerical stability of area computations
    ref = S["lp"][end[0] - 59]
    lpS = S["lp"] - ref
    lpF = F["lp"] - ref
    basis = lpF - lpS

    # ---------------- per-venue window features
    for tag, V, lp in (("s", S, lpS), ("f", F, lpF)):
        flow = 2 * V["bv"] - V["v"]
        cs_flow, cs_v, cs_n = _cs(flow), _cs(V["v"]), _cs(V["n"])
        r1 = np.diff(lp, prepend=lp[0])
        cs_r2 = _cs(r1 ** 2)
        for w in WINDOWS:
            f[f"{tag}_r{w}"] = lp[end] - lp[end - w]
            f[f"{tag}_flow{w}"] = _wsum(cs_flow, end, w)
            f[f"{tag}_vol{w}"] = _wsum(cs_v, end, w)
            if w >= 5:
                f[f"{tag}_n{w}"] = _wsum(cs_n, end, w)
            if w >= 10:
                f[f"{tag}_rv{w}"] = _wsum(cs_r2, end, w)
        # inactivity: seconds since last trade (staleness of the quote)
        traded = V["n"] > 0
        idx = np.where(traded, np.arange(len(traded)), -1)
        last_trade = np.maximum.accumulate(idx)
        f[f"{tag}_stale"] = (end - last_trade[end]).astype(float)
        # minute bar (OHLC from seconds of bar m)
        lo_, hi_ = end - 59, end
        hh = pd.Series(V["h"]).rolling(60, min_periods=1).max().to_numpy()
        ll = pd.Series(V["l"]).rolling(60, min_periods=1).min().to_numpy()
        f[f"{tag}_hi"] = np.log(hh[hi_]) - ref
        f[f"{tag}_lo"] = np.log(ll[hi_]) - ref
        f[f"{tag}_close"] = lp[end]
        f[f"{tag}_open"] = lp[lo_ - 1]
        # where inside the minute was the extreme reached (0=start,1=end)
        win = np.lib.stride_tricks.sliding_window_view(lp, 60)[lo_]
        f[f"{tag}_argmax"] = win.argmax(1) / 59.0
        f[f"{tag}_argmin"] = win.argmin(1) / 59.0
        # quote volume (for VWAP at minute level)
        f[f"{tag}_qv60"] = _wsum(_cs(V["qv"]), end, 60)

    # futures-only: large-trade flow and max print
    cs_l = _cs(F["lbv"] - F["lsv"])
    cs_lv = _cs(F["lbv"] + F["lsv"])
    for w in (10, 60, 300):
        f[f"f_lflow{w}"] = _wsum(cs_l, end, w)
        f[f"f_lvol{w}"] = _wsum(cs_lv, end, w)
    mx = pd.Series(F["mxq"]).rolling(60, min_periods=1).max().to_numpy()
    f["f_maxq60"] = mx[end]

    # ---------------- cross-venue basis (perp leads spot)
    cs_b = _cs(basis)
    f["basis"] = basis[end]
    for w in (5, 10, 30, 60, 120, 300, 600):
        f[f"basis_dev{w}"] = basis[end] - _wsum(cs_b, end, w) / w
    for w in (1, 2, 5, 10, 30, 60):
        f[f"basis_chg{w}"] = basis[end] - basis[end - w]

    # ---------------- Levy areas (signature level-2 antisymmetric part)
    cflowS = np.cumsum(2 * S["bv"] - S["v"])
    cflowF = np.cumsum(2 * F["bv"] - F["v"])
    for w in (30, 60, 300):
        f[f"levy_fs{w}"] = _levy(lpF, lpS, end, w)
        f[f"levy_sflow{w}"] = _levy(lpS, cflowS, end, w)
        f[f"levy_fflow{w}"] = _levy(lpF, cflowF, end, w)
        f[f"levy_fsflow{w}"] = _levy(cflowF, cflowS, end, w)

    # ---------------- ETH
    if E is not None:
        lpE = E["lp"] - E["lp"][PRE]
        flowE = _cs(2 * E["bv"] - E["v"])
        cs_vE = _cs(E["v"])
        for w in (5, 10, 30, 60, 300):
            f[f"e_r{w}"] = lpE[end] - lpE[end - w]
        for w in (60, 300):
            f[f"e_imb{w}"] = _wsum(flowE, end, w) / (_wsum(cs_vE, end, w) + 1e-12)
            f[f"levy_es{w}"] = _levy(lpE, lpS, end, w)
        f["e_close"] = E["lp"][end]
    # ---------------- forward prices for targets & latency analysis (NOT features)
    if forward:
        for d in LATENCY:
            f[f"px_s_d{d}"] = lpS[end + d]
            f[f"px_f_d{d}"] = lpF[end + d]
    out = pd.DataFrame(f)
    out["ref"] = ref
    return out


def build(start: dt.date, end: dt.date, workers: int = 4) -> pd.DataFrame:
    from multiprocessing import Pool

    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    with Pool(workers) as pool:
        parts = [p for p in pool.imap(day_features, days, chunksize=4) if p is not None]
    df = pd.concat(parts, ignore_index=True)
    # put all log-price levels on a common reference (each day was centered on its own ref)
    for c in [c for c in df.columns if c.endswith(("_close", "_open", "_hi", "_lo")) or c.startswith("px_")]:
        if c.startswith("e_") or c == "e_close":
            continue
        df[c] = df[c] + df["ref"]
    df = df.drop(columns=["ref"])
    float_cols = df.select_dtypes("float64").columns
    keep64 = [c for c in float_cols if c.endswith(("_close", "_open", "_hi", "_lo")) or c.startswith(("px_", "basis"))]
    df[[c for c in float_cols if c not in keep64]] = df[[c for c in float_cols if c not in keep64]].astype("float32")
    return df
