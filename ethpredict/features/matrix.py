"""Turn the raw panel into a stationary, scale-free model matrix.

Everything is expressed in units of the current volatility / typical volume, so a model trained in
a calm month transfers to a wild month (and vice versa).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .seconds import WINDOWS

EPS = 1e-12

# columns that are price/volume LEVELS (non-stationary) or bookkeeping -> never fed to models
LEVEL_COLS = {"ts", "contiguous", "s_close", "f_close", "s_open", "f_open", "s_hi", "s_lo", "f_hi", "f_lo",
              "e_close", "s_qv60", "f_qv60", "sigma", "basis", "hour", "prem_ma480", "dliq0.20"}


def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    sig = df["sigma"].astype(float)                       # per-minute vol (log-return units)
    out: dict[str, pd.Series] = {}
    for tag in ("s", "f"):
        vbase = df[f"{tag}_vol60"].astype(float).rolling(1440, min_periods=60).mean() / 60.0  # BTC / second
        for w in WINDOWS:
            sc = sig * np.sqrt(w / 60.0)
            out[f"{tag}_r{w}z"] = df[f"{tag}_r{w}"] / (sc + EPS)
            out[f"{tag}_imb{w}"] = df[f"{tag}_flow{w}"] / (df[f"{tag}_vol{w}"] + EPS)
            out[f"{tag}_flowz{w}"] = df[f"{tag}_flow{w}"] / (vbase * w + EPS)
            out[f"{tag}_lvol{w}"] = np.log(df[f"{tag}_vol{w}"] / (vbase * w + EPS) + 1e-3)
            if f"{tag}_n{w}" in df:
                out[f"{tag}_ln{w}"] = np.log1p(df[f"{tag}_n{w}"])
            if f"{tag}_rv{w}" in df:
                out[f"{tag}_lrv{w}"] = np.log(df[f"{tag}_rv{w}"] / (sig ** 2 * w / 60.0 + EPS) + 1e-6)
        out[f"{tag}_stale"] = np.log1p(df[f"{tag}_stale"])
        out[f"{tag}_argmax"] = df[f"{tag}_argmax"]
        out[f"{tag}_argmin"] = df[f"{tag}_argmin"]
        # scale-free versions of the raw vbase-normalized levy areas
        for w in (30, 60, 300):
            sc_p = sig * np.sqrt(w / 60.0)
            out[f"levy_{tag}flow{w}z"] = df[f"levy_{tag}flow{w}"] / (sc_p * vbase * w + EPS)
    vbf = df["f_vol60"].astype(float).rolling(1440, min_periods=60).mean() / 60.0
    vbs = df["s_vol60"].astype(float).rolling(1440, min_periods=60).mean() / 60.0
    for w in (10, 60, 300):
        out[f"f_lflowz{w}s"] = df[f"f_lflow{w}"] / (vbf * w + EPS)
        out[f"f_lshare{w}"] = df[f"f_lvol{w}"] / (df[f"f_vol{w}"] + EPS)
    for w in (30, 60, 300):
        sc = sig ** 2 * (w / 60.0)
        out[f"levy_fs{w}z"] = df[f"levy_fs{w}"] / (sc + EPS)
        out[f"levy_fsflow{w}z"] = df[f"levy_fsflow{w}"] / (vbf * vbs * w * w + EPS)
        # normalized by the actual path roughness inside the window (pure "who led" measure)
        out[f"levy_fs{w}n"] = df[f"levy_fs{w}"] / (np.sqrt(df[f"s_rv{w}"].astype(float) * df[f"f_rv{w}"].astype(float)) * w / 2 + EPS)
    for c in [c for c in df.columns if c.startswith("basis_dev") and not c.endswith("m")]:
        out[c + "z"] = df[c] / (sig + EPS)
    for c in [c for c in df.columns if c.startswith("basis_chg")]:
        out[c + "z"] = df[c] / (sig + EPS)
    if "e_r5" in df:
        for w in (5, 10, 30, 60, 300):
            out[f"e_r{w}z"] = df[f"e_r{w}"] / (sig * np.sqrt(w / 60.0) + EPS)
            out[f"be_r{w}z"] = (df[f"s_r{w}"] - df[f"e_r{w}"]) / (sig * np.sqrt(w / 60.0) + EPS)
        for w in (60, 300):
            out[f"e_imb{w}"] = df[f"e_imb{w}"]
            out[f"levy_es{w}z"] = df[f"levy_es{w}"] / (sig ** 2 * w / 60.0 + EPS)

    # minute-level features computed in panel.py are already normalized
    raw_second = {c for c in df.columns if c.split("_")[0] in ("s", "f", "e") and any(
        c.startswith(f"{t}_{k}") for t in ("s", "f", "e") for k in ("r", "flow", "vol", "n", "rv", "stale", "argm", "lflow", "lvol", "maxq", "imb"))}
    raw_second -= {c for c in df.columns if c.endswith("m") and ("imb" in c or "flowz" in c or "relvol" in c or "lflowz" in c)}
    skip = LEVEL_COLS | raw_second | {c for c in df.columns if c.startswith(("y_", "px_", "levy_", "basis_dev", "basis_chg"))}
    skip -= {c for c in df.columns if c.startswith("basis_dev") and c.endswith("m")}
    for c in df.columns:
        if c in skip or c in out:
            continue
        out[c] = df[c]
    X = pd.DataFrame(out, index=df.index).astype("float32")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """v2 additions motivated by the literature survey:
    * sign-based reversal kernels  sum_j j^-a * tanh(l * r_{t-j}/sigma)   (Kitron & Wengrowicz 2026)
    * reversal conditioned on whether the last bar was flow-driven
    * quarter-hour clock interactions (quarter-hour effect, 2026)
    """
    out = {}
    sig = df["sigma"].astype(float)
    for tag, col in (("s", "s_close"), ("f", "f_close")):
        r = df[col].diff() / (sig + EPS)
        th = np.tanh(1.0 * r)
        for a in (0.5, 1.0):
            acc = 0
            for j in range(1, 13):
                acc = acc + j ** (-a) * th.shift(j - 1)
            out[f"{tag}_revk{a}"] = acc
        # 5-minute bar version
        r5 = df[col].diff(5) / (sig * np.sqrt(5) + EPS)
        acc = 0
        for j in range(1, 7):
            acc = acc + j ** (-0.5) * np.tanh(r5.shift(5 * (j - 1)))
        out[f"{tag}_revk5m"] = acc
        imb = df[f"{tag}_flow60"] / (df[f"{tag}_vol60"] + EPS)
        driven = (np.sign(r) == np.sign(imb)).astype(float)
        out[f"{tag}_R1_driven"] = r * driven
        out[f"{tag}_R1_opposed"] = r * (1 - driven)
        r3 = df[col].diff(3) / (sig * np.sqrt(3) + EPS)
        imb3 = df[f"{tag}_flow60"].rolling(3).sum() / (df[f"{tag}_vol60"].rolling(3).sum() + EPS)
        out[f"{tag}_R3_x_imb3"] = r3 * imb3
    ts_fc = pd.DatetimeIndex(df["ts"]) + pd.Timedelta(minutes=1)
    qh = (ts_fc.minute % 15 == 0).astype(float)
    hr = (ts_fc.minute == 0).astype(float)
    r1 = df["s_close"].diff() / (sig + EPS)
    r5 = df["s_close"].diff(5) / (sig * np.sqrt(5) + EPS)
    out["qh_next"] = qh
    out["qh_x_R1"] = qh * r1
    out["qh_x_R5"] = qh * r5
    out["hr_x_R5"] = hr * r5
    # just after a quarter hour (minutes 1..3 into it)
    m15 = ts_fc.minute % 15
    out["qh_after"] = np.where(m15 <= 3, m15, -1).astype(float)
    m15 = np.asarray(m15)
    qa = np.zeros(len(df))
    for k in (1, 2, 3):
        qa = np.where(m15 == k, (df["s_close"].diff(k) / (sig * np.sqrt(k) + EPS)).values, qa)
    out["qh_after_x_R"] = qa
    return pd.DataFrame(out, index=df.index).astype("float32").replace([np.inf, -np.inf], np.nan)


def target(df: pd.DataFrame, h: int, venue: str = "s", delay: int | None = None) -> pd.Series:
    col = f"y_{venue}{h}" if delay is None else f"y_{venue}{h}_d{delay}"
    return df[col]


def target_scale(df: pd.DataFrame, h: int) -> pd.Series:
    return df["sigma"].astype(float) * np.sqrt(h)
