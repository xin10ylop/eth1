"""'Coinbase premium' features (Coinbase ETH-USD vs Binance ETH/USDT; product set in config.py).

Mechanism: US institutional / ETF-related spot demand executes on Coinbase first; cross-venue
arbitrage carries it to Binance with a lag. The USD-vs-USDT level drifts slowly, so only deviations
of the premium from its own recent mean are used. Top feature of every satellite model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def features(panel: pd.DataFrame, cb: pd.DataFrame) -> pd.DataFrame:
    """panel: needs ts, s_close (log), sigma. cb: Coinbase 1m candles (ts = open time)."""
    c = cb.set_index("ts")["close"].reindex(panel["ts"].values)
    v = cb.set_index("ts")["volume"].reindex(panel["ts"].values)
    lc = np.log(c.ffill().values)
    stale = c.isna().values                      # no Coinbase trade in that minute
    prem = lc - panel["s_close"].values
    sig = panel["sigma"].values
    P = pd.Series(prem)
    out = {}
    for k in (15, 60, 1440):
        out[f"cb_prem_dev{k}"] = (P - P.rolling(k, min_periods=5).mean()) / sig
    for k in (1, 5):
        out[f"cb_prem_chg{k}"] = P.diff(k) / sig
    L = pd.Series(lc)
    for k in (1, 5):
        out[f"cb_r{k}"] = L.diff(k) / (sig * np.sqrt(k))
    vv = pd.Series(v.fillna(0).values)
    out["cb_relvol15"] = np.log((vv.rolling(15, min_periods=1).mean() + 1e-6) / (vv.rolling(1440, min_periods=60).mean() + 1e-6))
    out["cb_share"] = np.log((vv.rolling(15, min_periods=1).sum() + 1e-6) / (panel["s_vol60"].astype(float).rolling(15, min_periods=1).sum().values + 1e-6))
    out["cb_stale"] = stale.astype(float)
    return pd.DataFrame(out, index=panel.index).astype("float32").replace([np.inf, -np.inf], np.nan)
