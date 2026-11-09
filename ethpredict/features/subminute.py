"""Sub-minute targets: price change over the next 10 and 15 SECONDS after the bar close.

  y_{v}{H}s       = log P(T+H) - log P(T)                  (forecast instant T = bar close)
  y_{v}{H}s_d{D}  = log P(T+D+H) - log P(T+D)              (you receive the forecast D seconds late
                                                            and hold for H seconds from then)
P(t) = last trade price before t (per-second closes, forward-filled). v in {s: spot, f: perp}.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..config import DIR_PERP, DIR_SPOT, FEAT, SEC

HS = (10, 15)
DELAYS = (1, 2, 5)
MAXOFF = max(HS) + max(DELAYS) + 1


def _closes(venue: str, day: dt.date) -> np.ndarray | None:
    p = SEC / venue / f"{day}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p, columns=["c"])["c"].to_numpy(float)


def day_targets(day: dt.date) -> pd.DataFrame | None:
    out = {"ts": (pd.Timestamp(day) + pd.to_timedelta(60 * np.arange(1440), unit="s")).values}
    for v, venue in (("s", DIR_SPOT), ("f", DIR_PERP)):
        cur = _closes(venue, day)
        if cur is None:
            return None
        nxt = _closes(venue, day + dt.timedelta(days=1))
        tail = nxt[:MAXOFF] if nxt is not None else np.full(MAXOFF, np.nan)
        c = np.concatenate([cur, tail])
        lp = np.log(pd.Series(c).ffill().to_numpy())
        end = 60 * np.arange(1440) + 59            # last second of bar m; P(T) = lp[end]
        for H in HS:
            out[f"y_{v}{H}s"] = lp[end + H] - lp[end]
            for D in DELAYS:
                out[f"y_{v}{H}s_d{D}"] = lp[end + D + H] - lp[end + D]
    return pd.DataFrame(out)


def build(start: str, end: str) -> pd.DataFrame:
    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    parts = [day_targets(s + dt.timedelta(days=i)) for i in range((e - s).days + 1)]
    df = pd.concat([p for p in parts if p is not None], ignore_index=True)
    return df


if __name__ == "__main__":
    import sys

    panel = pd.read_parquet(FEAT / "panel.parquet", columns=["ts"])
    panel = panel[panel["ts"] >= "2025-05-03"].reset_index(drop=True)
    d = build(sys.argv[1] if len(sys.argv) > 1 else "2025-05-01", sys.argv[2] if len(sys.argv) > 2 else "2026-11-02")
    d = panel.merge(d, on="ts", how="left")
    d.drop(columns="ts").astype("float32").to_parquet(FEAT / "Ysub.parquet")
    print(d.shape, d.isna().mean().round(4).to_dict())
