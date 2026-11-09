"""Minute-level panel: merges second-level features with longer-window features, derivatives
data and targets.

Row m = one-minute bar whose open time is `ts`. The forecast is issued at ts + 60s (bar close)
and every feature uses information available at that instant only. Targets:

    y_s{h}  = log P_spot(close of bar m+h) - log P_spot(close of bar m)        h = 1..5
    y_f{h}  = same for the perpetual future
    y_s{h}_d{D} = target measured from D seconds AFTER the forecast instant (latency analysis)
"""
from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import pandas as pd

from . import seconds as secfeat
from ..config import DIR_DEPTH, DIR_METRICS, DIR_PREMIUM, FEAT, HORIZONS, SEC

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

EPS = 1e-12


def _read_dir(name: str | None) -> pd.DataFrame | None:
    if name is None:
        return None
    files = sorted((SEC / name).glob("*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _roll_sum(x: pd.Series, k: int) -> pd.Series:
    return x.rolling(k, min_periods=1).sum()


def add_minute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    c = df["s_close"]
    cf = df["f_close"]
    r1 = c.diff()
    rf1 = cf.diff()

    # ---------------- volatility state (past only)
    rv = df["s_rv60"].astype(float)                       # realized var of the bar from 1s returns
    rv_f = df["f_rv60"].astype(float)
    sig2 = rv.ewm(span=30, min_periods=5).mean()          # per-minute variance estimate
    sig2_slow = rv.ewm(span=720, min_periods=30).mean()
    sig = np.sqrt(sig2.clip(lower=1e-10))
    out["sigma"] = sig
    out["lsig"] = np.log(sig)
    out["vol_ratio_fast_slow"] = np.log((sig2 + EPS) / (sig2_slow + EPS))
    for k in (5, 60, 240):
        out[f"lrv{k}"] = np.log(_roll_sum(rv, k) / k + EPS) - np.log(sig2_slow + EPS)
    out["rv_f_over_s"] = np.log((rv_f.rolling(15, min_periods=1).mean() + EPS) / (rv.rolling(15, min_periods=1).mean() + EPS))
    # minute-return based variance vs 1s-based (microstructure noise / mean-reversion indicator)
    out["vr_1s_vs_1m"] = np.log((r1 ** 2).ewm(span=60).mean() + EPS) - np.log(rv.ewm(span=60).mean() + EPS)
    # variance ratio over 5 minutes (trend vs mean reversion regime)
    r5 = c.diff(5)
    out["vr5_240"] = np.log((r5 ** 2).rolling(240, min_periods=30).mean() + EPS) - np.log(5 * (r1 ** 2).rolling(240, min_periods=30).mean() + EPS)
    # rolling first-order autocorrelation of minute returns (regime of momentum/reversal)
    for k in (60, 240):
        out[f"ac1_{k}"] = r1.rolling(k, min_periods=20).corr(r1.shift(1))

    # ---------------- multi-minute returns, vol-normalized
    for k in (1, 2, 3, 5, 10, 15, 30, 60, 120, 240, 720, 1440):
        out[f"R{k}"] = (c - c.shift(k)) / (sig * np.sqrt(k))
    for k in (1, 5, 15, 60):
        out[f"RF{k}"] = (cf - cf.shift(k)) / (sig * np.sqrt(k))

    # ---------------- order flow over minutes
    for tag in ("s", "f"):
        fl = df[f"{tag}_flow60"].astype(float)
        vo = df[f"{tag}_vol60"].astype(float)
        vbase = vo.rolling(1440, min_periods=60).mean()
        for k in (1, 3, 5, 15, 60, 240):
            sf = _roll_sum(fl, k)
            sv = _roll_sum(vo, k)
            out[f"{tag}_imb{k}m"] = sf / (sv + EPS)
            out[f"{tag}_flowz{k}m"] = sf / (vbase * k + EPS)
        for k in (5, 60):
            out[f"{tag}_relvol{k}m"] = np.log(_roll_sum(vo, k) / k + EPS) - np.log(vbase + EPS)
        n = df[f"{tag}_n60"].astype(float)
        out[f"{tag}_avgsize"] = np.log(_roll_sum(vo, 5) / (_roll_sum(n, 5) + 1) + EPS) - np.log(vo.rolling(1440, min_periods=60).sum() / (n.rolling(1440, min_periods=60).sum() + 1) + EPS)
    # large prints
    lv = df["f_lflow60"].astype(float)
    vb = df["f_vol60"].astype(float).rolling(1440, min_periods=60).mean()
    for k in (1, 5, 15):
        out[f"f_lflowz{k}m"] = _roll_sum(lv, k) / (vb * k + EPS)
    out["f_maxq_rel"] = np.log(df["f_maxq60"].astype(float) + 1e-3) - np.log(df["f_maxq60"].astype(float).rolling(1440, min_periods=60).median() + 1e-3)

    # ---------------- Kyle lambda & absorption (flow that did not move price)
    flz = df["f_flow60"].astype(float) / (vb + EPS)
    cov = (rf1 * flz).rolling(240, min_periods=60).mean() - rf1.rolling(240, min_periods=60).mean() * flz.rolling(240, min_periods=60).mean()
    var = flz.rolling(240, min_periods=60).var()
    lam = cov / (var + EPS)
    out["kyle_lambda"] = lam / (sig + EPS)
    for k in (1, 5):
        resid = rf1.rolling(k, min_periods=1).sum() - lam * flz.rolling(k, min_periods=1).sum()
        out[f"absorb{k}m"] = resid / (sig * np.sqrt(k) + EPS)
    out["flow_ret_corr240"] = rf1.rolling(240, min_periods=60).corr(flz)

    # ---------------- price location
    hi = df["s_hi"].astype(float); lo = df["s_lo"].astype(float)
    for k in (15, 60, 240, 1440):
        H = hi.rolling(k, min_periods=1).max(); L = lo.rolling(k, min_periods=1).min()
        out[f"rngpos{k}"] = (c - L) / (H - L + EPS) - 0.5
        out[f"rngwidth{k}"] = (H - L) / (sig * np.sqrt(k) + EPS)
    for k in (20, 60, 240):
        mu = c.rolling(k, min_periods=5).mean(); sd = c.rolling(k, min_periods=5).std()
        out[f"bbz{k}"] = (c - mu) / (sd + EPS)
    qv = df["s_qv60"].astype(float); vo = df["s_vol60"].astype(float)
    for k in (15, 60, 240):
        vwap = np.log(_roll_sum(qv, k) / (_roll_sum(vo, k) + EPS) + EPS)
        out[f"vwapdev{k}"] = (c - vwap) / (sig * np.sqrt(k) + EPS)
    px = np.exp(c)
    for step in (100.0, 500.0, 1000.0):
        frac = (px % step) / step
        out[f"round{int(step)}"] = frac - 0.5
        out[f"round{int(step)}_dist"] = np.minimum(px % step, step - px % step) / (px * sig + EPS)
    out["candle_body"] = (c - df["s_open"]) / (hi - lo + EPS)
    out["upper_wick"] = (hi - np.maximum(c, df["s_open"])) / (sig + EPS)
    out["lower_wick"] = (np.minimum(c, df["s_open"]) - lo) / (sig + EPS)

    # ---------------- basis (longer windows)
    b = df["basis"]
    for k in (15, 60, 240, 1440):
        mu = b.rolling(k, min_periods=5).mean()
        out[f"basis_dev{k}m"] = (b - mu) / (sig + EPS)
    out["basis_lvl"] = b
    out["basis_z1440"] = (b - b.rolling(1440, min_periods=60).mean()) / (b.rolling(1440, min_periods=60).std() + EPS)

    # ---------------- relative value vs the other major (BTC): E = its return, BE = ours minus its
    if "e_close" in df:
        e = df["e_close"]
        for k in (1, 5, 15, 60, 240):
            out[f"E{k}"] = (e - e.shift(k)) / (sig * np.sqrt(k))
            out[f"BE{k}"] = ((c - c.shift(k)) - (e - e.shift(k))) / (sig * np.sqrt(k))
        er1 = e.diff()
        cov = (r1 * er1).rolling(240, min_periods=60).mean() - r1.rolling(240, min_periods=60).mean() * er1.rolling(240, min_periods=60).mean()
        beta = cov / (er1.rolling(240, min_periods=60).var() + EPS)
        out["x_beta"] = beta
        out["x_resid5"] = ((c - c.shift(5)) - beta * (e - e.shift(5))) / (sig * np.sqrt(5))

    # ---------------- calendar
    ts = pd.DatetimeIndex(df["ts"]) + pd.Timedelta(minutes=1)   # forecast instant
    mod = ts.hour * 60 + ts.minute
    out["tod_sin"] = np.sin(2 * np.pi * mod / 1440); out["tod_cos"] = np.cos(2 * np.pi * mod / 1440)
    out["hour"] = ts.hour.astype(float)
    out["minute"] = ts.minute.astype(float)
    out["mod5"] = (ts.minute % 5).astype(float)
    out["mod15"] = (ts.minute % 15).astype(float)
    out["dow"] = ts.dayofweek.astype(float)
    out["to_funding"] = ((480 - (mod % 480)) % 480).astype(float)
    df = pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)
    return df


def add_aux(df: pd.DataFrame) -> pd.DataFrame:
    """Merge futures metrics, book depth and premium index -- all strictly as-of the forecast instant."""
    # all timestamps handled as tz-naive UTC datetime64[ns]
    t_fc = pd.DatetimeIndex(df["ts"]).tz_localize(None) + pd.Timedelta(minutes=1)
    base = pd.DataFrame({"t": t_fc.astype("datetime64[ns]")}, index=df.index).reset_index()

    df = df.copy()
    dep = _read_dir(DIR_DEPTH)
    if dep is not None:
        dep = dep.sort_values("timestamp").drop_duplicates("timestamp")
        dep["timestamp"] = pd.to_datetime(dep["timestamp"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
        a = {}
        for p in ("0.20", "1.00", "2.00", "5.00"):
            bid, ask = dep[f"d-{p}"], dep[f"d+{p}"]
            a[f"dimb{p}"] = (bid - ask) / (bid + ask)
        a["dimb_near_vs_far"] = a["dimb0.20"] - a["dimb5.00"]
        a["dliq0.20"] = np.log(dep["d-0.20"] + dep["d+0.20"])
        dd = pd.DataFrame(a)
        dd["timestamp"] = dep["timestamp"].values
        # snapshot must be at least 1s old at forecast time
        base2 = base.copy(); base2["t"] = base2["t"] - pd.Timedelta(seconds=1)
        m = pd.merge_asof(base2.sort_values("t"), dd.sort_values("timestamp"), left_on="t", right_on="timestamp",
                          direction="backward", tolerance=pd.Timedelta(minutes=3))
        m = m.set_index("index").sort_index()
        for col in dd.columns:
            if col == "timestamp":
                continue
            df[col] = m[col].astype("float32")
        for col in ("dimb0.20", "dimb1.00"):
            df[f"{col}_chg5"] = (df[col] - df[col].shift(5)).astype("float32")
            df[f"{col}_ma15"] = df[col].rolling(15, min_periods=1).mean().astype("float32")
        df["dliq_rel"] = (df["dliq0.20"] - df["dliq0.20"].rolling(1440, min_periods=60).mean()).astype("float32")

    met = _read_dir(DIR_METRICS)
    if met is not None:
        met["t"] = pd.to_datetime(met["create_time"]).astype("datetime64[ns]")
        met = met.sort_values("t").drop_duplicates("t")
        mm = pd.DataFrame({
            "t": met["t"].values,
            "oi": np.log(met["sum_open_interest"].astype(float)).values,
            "ls_top": np.log(met["sum_toptrader_long_short_ratio"].astype(float)).values,
            "ls_cnt": np.log(met["count_long_short_ratio"].astype(float)).values,
            "taker_ls": np.log(met["sum_taker_long_short_vol_ratio"].astype(float).clip(lower=1e-3)).values,
        })
        for k in (1, 3, 12):
            mm[f"oi_chg{k}"] = mm["oi"].diff(k)
        mm["ls_top_chg"] = mm["ls_top"].diff(3)
        # metric rows are stamped at period end; demand 5 min of publication slack
        base2 = base.copy(); base2["t"] = base2["t"] - pd.Timedelta(minutes=5)
        m = pd.merge_asof(base2.sort_values("t"), mm, on="t", direction="backward", tolerance=pd.Timedelta(minutes=30))
        m = m.set_index("index").sort_index()
        for col in ("oi_chg1", "oi_chg3", "oi_chg12", "ls_top", "ls_cnt", "taker_ls", "ls_top_chg"):
            df[col] = m[col].astype("float32")

    pre = _read_dir(DIR_PREMIUM)
    if pre is not None:
        pre["t"] = pd.to_datetime(pre["open_time"].astype("int64"), unit="ms").astype("datetime64[ns]") + pd.Timedelta(minutes=1)
        pre = pre.sort_values("t").drop_duplicates("t")
        pp = pd.DataFrame({"t": pre["t"].values, "prem": pre["close"].astype(float).values})
        m = pd.merge_asof(base.sort_values("t"), pp, on="t", direction="backward", tolerance=pd.Timedelta(minutes=2))
        m = m.set_index("index").sort_index()
        df["prem"] = m["prem"].astype("float32")
        df["prem_ma480"] = df["prem"].rolling(480, min_periods=30).mean().astype("float32")
        df["prem_dev60"] = (df["prem"] - df["prem"].rolling(60, min_periods=5).mean()).astype("float32")
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["s_close"]; cf = df["f_close"]
    ts = pd.DatetimeIndex(df["ts"])
    contiguous = pd.Series(ts).diff().dt.total_seconds().fillna(60).eq(60).values
    for h in HORIZONS:
        ok = pd.Series(ts).shift(-h).sub(pd.Series(ts)).dt.total_seconds().eq(60 * h).values
        df[f"y_s{h}"] = np.where(ok, c.shift(-h) - c, np.nan)
        df[f"y_f{h}"] = np.where(ok, cf.shift(-h) - cf, np.nan)
        for d in secfeat.LATENCY:
            df[f"y_s{h}_d{d}"] = np.where(ok, c.shift(-h) - df[f"px_s_d{d}"], np.nan)
            df[f"y_f{h}_d{d}"] = np.where(ok, cf.shift(-h) - df[f"px_f_d{d}"], np.nan)
    df["contiguous"] = contiguous
    return df.copy()


def build(start: str, end: str, workers: int = 4, out_name: str = "panel") -> pd.DataFrame:
    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    df = secfeat.build(s, e, workers)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    df = add_minute_features(df)
    df = add_aux(df)
    df = add_targets(df)
    path = FEAT / f"{out_name}.parquet"
    df.to_parquet(path, index=False)
    return df


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-05-01")
    ap.add_argument("--end", default="2026-11-02")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="panel")
    a = ap.parse_args()
    t = time.time()
    d = build(a.start, a.end, a.workers, a.name)
    print(d.shape, f"{time.time()-t:.0f}s")
