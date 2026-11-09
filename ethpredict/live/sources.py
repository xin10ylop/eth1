"""Live feature construction -- the SAME code path as training (secfeat / altfeat / panel / features).

Two interchangeable data sources:
  RestSource   -- official public Binance endpoints (spot mirror data-api.binance.vision and
                  USD-M futures fapi.binance.com).  Futures endpoints are geo-restricted in some
                  jurisdictions; use the spot-only model there.
  ReplaySource -- reads the historical per-second files; used to prove that live features are
                  identical to the training features (scripts/live/check_live_parity.py).

Window layout at forecast instant T (a minute boundary):
  * minute history: HIST_MIN minutes before T (needed for 24h rolling statistics)
  * second-level detail: SEC_WIN seconds before T for every venue
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from ..config import (COINBASE_FILE, COINBASE_PRODUCT, DATA_DIRS, DIR_LEAD_PERP, DIR_LEAD_SPOT, DIR_PERP, DIR_PREMIUM,
                      LEAD_KEY, LEAD_PERP_SYMBOL, LEAD_SPOT_SYMBOL, PERP_SYMBOL, SPOT_SYMBOLS, TARGET_KEY)
from ..features import crossasset as altfeat
from ..features import matrix as features
from ..features import panel
from ..features import seconds as secfeat

HIST_MIN = 2600        # minutes of minute-level history (slow EWMs converge to <0.1%)
SEC_WIN = 1260         # seconds of per-second detail (>= PRE + 60 for alt vol estimates, + slack)
SPOT_API = ["https://data-api.binance.vision/api/v3", "https://api.binance.com/api/v3"]
FUT_API = "https://fapi.binance.com"
SEC_COLS = ["o", "h", "l", "c", "v", "qv", "n", "bv"]


def _grid(t0_ms: int, n: int) -> pd.DataFrame:
    return pd.DataFrame(index=np.arange(n))


def kl_to_seconds(rows: list, t0_ms: int, n: int) -> pd.DataFrame:
    """Binance 1s klines -> dense per-second frame of length n starting at t0_ms."""
    out = {k: np.full(n, np.nan) for k in ("o", "h", "l", "c")}
    out.update({k: np.zeros(n) for k in ("v", "qv", "n", "bv")})
    if rows:
        a = np.array([[float(x) for x in r[:11]] for r in rows])
        sec = ((a[:, 0] - t0_ms) // 1000).astype(int)
        ok = (sec >= 0) & (sec < n)
        sec, a = sec[ok], a[ok]
        for j, k in ((1, "o"), (2, "h"), (3, "l"), (4, "c"), (5, "v"), (7, "qv"), (8, "n"), (9, "bv")):
            out[k][sec] = a[:, j]
        nov = out["n"] == 0
        for k in ("o", "h", "l", "c"):
            out[k][nov] = np.nan
    return pd.DataFrame(out)


def minute_to_seconds(m: pd.DataFrame, t0_ms: int, n: int) -> pd.DataFrame:
    """Embed 1-minute bars into a per-second grid (all activity on the minute's last second).
    Exact for every per-minute aggregate used by the minute-level features (close, high, low,
    volume, taker-buy volume, quote volume); NOT used for intra-minute features."""
    out = {k: np.full(n, np.nan) for k in ("o", "h", "l", "c")}
    out.update({k: np.zeros(n) for k in ("v", "qv", "n", "bv", "nb", "lbv", "lsv", "mxq")})
    if len(m):
        last = ((m["open_ms"].values - t0_ms) // 1000).astype(int) + 59
        ok = (last >= 0) & (last < n)
        last = last[ok]
        mm = m[ok]
        out["h"][last] = mm["h"].values
        out["l"][last] = mm["l"].values
        out["o"][last] = mm["c"].values
        out["c"][last] = mm["c"].values
        for k in ("v", "qv", "n", "bv"):
            out[k][last] = mm[k].fillna(0.0).values
    return pd.DataFrame(out)


# =============================================================================================
class ReplaySource:
    """Historical files -> the inputs a live process would have had at instant T."""

    def __init__(self):
        from ..config import SEC
        self.SEC = SEC
        self._cache: dict = {}

    def _day(self, venue, day):
        key = (venue, day)
        if key not in self._cache:
            p = self.SEC / venue / f"{day}.parquet"
            self._cache[key] = pd.read_parquet(p) if p.exists() else None
        return self._cache[key]

    def _range(self, venue, t0_ms, n):
        t0 = pd.Timestamp(t0_ms, unit="ms")
        days = pd.date_range(t0.normalize(), (t0 + pd.Timedelta(seconds=n)).normalize(), freq="D")
        parts = []
        for d in days:
            x = self._day(venue, d.date())
            if x is None:
                x = pd.DataFrame({c: (np.full(86400, np.nan) if c in "ohlc" else np.zeros(86400)) for c in SEC_COLS})
            parts.append(x)
        x = pd.concat(parts, ignore_index=True)
        off = int((t0 - days[0]).total_seconds())
        return x.iloc[off:off + n].reset_index(drop=True)

    def spot_seconds(self, symbol_key, t0_ms, n):
        return self._range(DATA_DIRS[symbol_key], t0_ms, n)

    def fut_seconds(self, t0_ms, n, symbol=PERP_SYMBOL):
        return self._range({PERP_SYMBOL: DIR_PERP, LEAD_PERP_SYMBOL: DIR_LEAD_PERP}[symbol], t0_ms, n)

    def minutes(self, role, t0_ms, n_min):
        """1-minute bars aggregated from the per-second history (= what the kline endpoint returns).
        role: "perp" (target perpetual) or "lead" (the other major's spot)."""
        x = self._range({"perp": DIR_PERP, "lead": DIR_LEAD_SPOT}[role], t0_ms, n_min * 60)
        g = np.arange(len(x)) // 60
        c = pd.Series(x["c"].values).groupby(g).last()
        out = pd.DataFrame({
            "open_ms": t0_ms + 60000 * np.arange(n_min),
            "c": x["c"].groupby(g).last().values, "h": x["h"].groupby(g).max().values, "l": x["l"].groupby(g).min().values,
            "v": x["v"].groupby(g).sum().values, "qv": x["qv"].groupby(g).sum().values, "n": x["n"].groupby(g).sum().values,
            "bv": x["bv"].groupby(g).sum().values})
        out["c"] = pd.Series(out["c"]).ffill().values
        return out

    def coinbase_minutes(self, t0_ms, n_min):
        if "cb" not in self._cache:
            p = self.SEC / "coinbase" / COINBASE_FILE
            self._cache["cb"] = pd.read_parquet(p) if p.exists() else None
        cb = self._cache["cb"]
        if cb is None:
            return None
        t0 = pd.Timestamp(t0_ms, unit="ms")
        return cb[(cb["ts"] >= t0) & (cb["ts"] < t0 + pd.Timedelta(minutes=n_min))].reset_index(drop=True)

    def premium(self, t0_ms, n_min):
        days = pd.date_range(pd.Timestamp(t0_ms, unit="ms").normalize(), pd.Timestamp(t0_ms + n_min * 60000, unit="ms").normalize())
        parts = [self._day(DIR_PREMIUM, d.date()) for d in days]
        p = pd.concat([x for x in parts if x is not None])
        p = p[(p["open_time"] >= t0_ms) & (p["open_time"] < t0_ms + n_min * 60000)]
        return pd.DataFrame({"open_ms": p["open_time"].values, "prem": p["close"].astype(float).values})


# =============================================================================================
class RestSource:
    def __init__(self, use_futures: bool = True, timeout: float = 10.0):
        self.s = requests.Session()
        self.timeout = timeout
        self.use_futures = use_futures

    def _get(self, bases, path, params):
        bases = bases if isinstance(bases, list) else [bases]
        err = None
        for b in bases:
            for k in range(3):
                try:
                    r = self.s.get(b + path, params=params, timeout=self.timeout)
                    if r.status_code == 200:
                        return r.json()
                    err = f"{r.status_code} {r.text[:200]}"
                    if r.status_code in (451, 403):
                        break
                except Exception as e:  # noqa: BLE001
                    err = str(e)
                time.sleep(0.3 * (k + 1))
        raise RuntimeError(f"GET {path} failed: {err}")

    def _klines(self, bases, path, symbol, interval, t0_ms, t1_ms, limit):
        rows, start = [], t0_ms
        step = {"1s": 1000, "1m": 60000}[interval]
        while start < t1_ms:
            r = self._get(bases, path, {"symbol": symbol, "interval": interval, "startTime": start,
                                        "endTime": t1_ms - 1, "limit": limit})
            if not r:
                break
            rows += r
            start = int(r[-1][0]) + step
            if len(r) < limit:
                break
        return rows

    def spot_seconds(self, symbol_key, t0_ms, n):
        sym = SPOT_SYMBOLS[symbol_key]
        rows = self._klines(SPOT_API, "/klines", sym, "1s", t0_ms, t0_ms + n * 1000, 1000)
        return kl_to_seconds(rows, t0_ms, n)

    def minutes(self, role, t0_ms, n_min):
        if role == "perp":
            rows = self._klines(FUT_API, "/fapi/v1/klines", PERP_SYMBOL, "1m", t0_ms, t0_ms + n_min * 60000, 1500)
        else:
            rows = self._klines(SPOT_API, "/klines", LEAD_SPOT_SYMBOL, "1m", t0_ms, t0_ms + n_min * 60000, 1000)
        a = np.array([[float(x) for x in r[:11]] for r in rows])
        return pd.DataFrame({"open_ms": a[:, 0].astype(np.int64), "c": a[:, 4], "h": a[:, 2], "l": a[:, 3], "v": a[:, 5],
                             "qv": a[:, 7], "n": a[:, 8], "bv": a[:, 9]})

    def fut_seconds(self, t0_ms, n, symbol=PERP_SYMBOL):
        """Per-second perp bars from aggTrades (same aggregation as the training data)."""
        trades, start, end = [], t0_ms, t0_ms + n * 1000
        r = self._get(FUT_API, "/fapi/v1/aggTrades", {"symbol": symbol, "startTime": start,
                                                       "endTime": min(end - 1, start + 3_599_000), "limit": 1000})
        trades += r
        while r and len(r) == 1000 and r[-1]["T"] < end:
            r = self._get(FUT_API, "/fapi/v1/aggTrades", {"symbol": symbol, "fromId": r[-1]["a"] + 1, "limit": 1000})
            trades += [x for x in r if x["T"] < end]
        if not trades:
            return secfeat_empty(n)
        t = np.array([x["T"] for x in trades], dtype=np.int64) - t0_ms
        p = np.array([float(x["p"]) for x in trades]); q = np.array([float(x["q"]) for x in trades])
        m = np.array([x["m"] for x in trades], dtype=bool)
        from ..data.binance import trades_to_1s
        return trades_to_1s(t, p, q, m, n)

    CB_URL = f"https://api.exchange.coinbase.com/products/{COINBASE_PRODUCT}"
    CB_FRESH_MIN = 15      # recent candles are revised after the fact -> rebuild these minutes from trades

    def coinbase_candles(self, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
        rows, t = [], t0
        while t < t1:
            t2 = min(t + pd.Timedelta(minutes=300), t1)
            r = self.s.get(self.CB_URL + "/candles", params={"granularity": 60, "start": t.isoformat(),
                                                             "end": (t2 - pd.Timedelta(seconds=1)).isoformat()}, timeout=self.timeout)
            if r.status_code == 200:
                rows += r.json()
            t = t2
        df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"]).drop_duplicates("time")
        df["ts"] = pd.to_datetime(df["time"], unit="s")
        return df[(df["ts"] >= t0) & (df["ts"] < t1)][["ts", "close", "volume"]].sort_values("ts").reset_index(drop=True)

    def coinbase_trades(self, min_time: pd.Timestamp, stop_id: int | None = None) -> pd.DataFrame:
        """Trades newer than min_time (and newer than stop_id), paging back with the CB-AFTER cursor."""
        out, after = [], None
        for _ in range(60):
            params = {"limit": 1000} if after is None else {"limit": 1000, "after": after}
            r = self.s.get(self.CB_URL + "/trades", params=params, timeout=self.timeout)
            r.raise_for_status()
            page = r.json()
            if not page:
                break
            out += page
            after = r.headers.get("CB-AFTER") or page[-1]["trade_id"]
            oldest = pd.Timestamp(page[-1]["time"]).tz_localize(None) if pd.Timestamp(page[-1]["time"]).tzinfo is None \
                else pd.Timestamp(page[-1]["time"]).tz_convert(None)
            if oldest < min_time or (stop_id is not None and page[-1]["trade_id"] <= stop_id):
                break
        if not out:
            return pd.DataFrame(columns=["trade_id", "t", "price", "size"])
        tt = pd.DataFrame(out)
        tt["t"] = pd.to_datetime(tt["time"], utc=True).dt.tz_localize(None)
        tt = tt.astype({"price": float, "size": float})[["trade_id", "t", "price", "size"]]
        if stop_id is not None:
            tt = tt[tt["trade_id"] > stop_id]
        return tt[tt["t"] >= min_time].drop_duplicates("trade_id").sort_values("trade_id").reset_index(drop=True)

    @staticmethod
    def trades_to_minutes(tt: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
        tt = tt[(tt["t"] >= t0) & (tt["t"] < t1)].sort_values("trade_id")
        g = tt.groupby(tt["t"].dt.floor("min"))
        m = pd.DataFrame({"close": g["price"].last(), "volume": g["size"].sum()})
        m.index.name = "ts"
        return m.reset_index()

    def coinbase_minutes(self, t0_ms, n_min):
        """Coinbase USD-product minutes: final candles for older minutes, the last CB_FRESH_MIN minutes
        rebuilt from raw trades (identical to the final candles once complete; verified)."""
        t0 = pd.Timestamp(t0_ms, unit="ms")
        stop = t0 + pd.Timedelta(minutes=n_min)
        cut = max(t0, stop - pd.Timedelta(minutes=self.CB_FRESH_MIN))
        old = self.coinbase_candles(t0, cut)
        tt = self.coinbase_trades(cut)
        for _ in range(8):                      # completeness barrier (see CachedSource._coinbase)
            if len(tt) and tt["t"].max() >= stop:
                break
            time.sleep(0.25)
            tt = self.coinbase_trades(cut)
        recent = self.trades_to_minutes(tt, cut, stop)
        return pd.concat([old, recent], ignore_index=True).sort_values("ts").reset_index(drop=True)

    def premium(self, t0_ms, n_min):
        rows = self._klines(FUT_API, "/fapi/v1/premiumIndexKlines", PERP_SYMBOL, "1m", t0_ms, t0_ms + n_min * 60000, 1500)
        return pd.DataFrame({"open_ms": [int(r[0]) for r in rows], "prem": [float(r[4]) for r in rows]})


def secfeat_empty(n):
    d = {k: np.full(n, np.nan) for k in ("o", "h", "l", "c")}
    d.update({k: np.zeros(n) for k in ("v", "qv", "n", "bv", "nb", "lbv", "lsv", "mxq")})
    return pd.DataFrame(d)


# =============================================================================================
def _arrays(df: pd.DataFrame, cols) -> dict:
    out = {c: df[c].to_numpy(np.float64) for c in cols if c in df}
    c = pd.Series(df["c"].to_numpy(np.float64)).ffill().bfill().to_numpy()
    out["lp"] = np.log(c)
    return out


def build_features(src, T: pd.Timestamp, use_futures: bool = True, last_n: int = 1, wide: bool = True) -> pd.DataFrame:
    """Full feature frame (same columns as training) for the `last_n` bars ending at instant T.

    wide=False skips the research-only inputs the ETH production models do not use (17-coin swarm
    `swarm2_*`, other books `book_*`): 13 fewer requests per minute. Their columns are then absent,
    so a model that needs them fails loudly in Predictor.raw instead of silently getting NaN."""
    T = pd.Timestamp(T).floor("min")
    T_ms = int(T.value // 10**6)
    h0_ms = T_ms - HIST_MIN * 60000
    nS = HIST_MIN * 60
    s0_ms = T_ms - SEC_WIN * 1000
    if isinstance(src, CachedSource):
        _prefetch(src, h0_ms, nS, s0_ms, use_futures, wide)
    # target spot: per-second for the whole history (exact realized variance per minute)
    S = src.spot_seconds(TARGET_KEY, h0_ms, nS)
    # perp: minute bars for history + real per-second bars for the recent window
    if use_futures:
        Fm = src.minutes("perp", h0_ms, HIST_MIN)
        F = minute_to_seconds(Fm, h0_ms, nS)
        Fs = src.fut_seconds(s0_ms, SEC_WIN)
        F.iloc[nS - SEC_WIN:] = Fs[F.columns].values
    else:
        F = S.copy()
        for k in ("nb", "lbv", "lsv", "mxq"):
            F[k] = 0.0
    Em = src.minutes("lead", h0_ms, HIST_MIN)
    E = minute_to_seconds(Em, h0_ms, nS)
    Es = src.spot_seconds(LEAD_KEY, s0_ms, SEC_WIN)
    E.iloc[nS - SEC_WIN:, [E.columns.get_loc(c) for c in SEC_COLS]] = Es[SEC_COLS].values
    m = np.arange(HIST_MIN)
    end = 60 * m + 59
    keep = end - 59 >= 660            # need 660s of history inside the window for the second-level windows
    m, end = m[keep], end[keep]
    ts = (pd.Timestamp(h0_ms, unit="ms") + pd.to_timedelta(60 * m, unit="s")).values
    df = secfeat.features_from_arrays(_arrays(S, SEC_COLS), _arrays(F, list(F.columns)), _arrays(E, ["c", "v", "bv"]),
                                      end, ts)
    for c in [c for c in df.columns if c.endswith(("_close", "_open", "_hi", "_lo")) and not c.startswith("e_")]:
        df[c] = df[c] + df["ref"]
    df = df.drop(columns=["ref"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = panel.add_minute_features(df)
    if use_futures:
        P = src.premium(h0_ms - 60000 * 600, HIST_MIN + 600)
        pts = pd.to_datetime(P["open_ms"], unit="ms")
        pr = pd.Series(P["prem"].values, index=pts).reindex(df["ts"])
        df["prem"] = pr.values.astype("float32")
        df["prem_ma480"] = df["prem"].rolling(480, min_periods=30).mean()
        df["prem_dev60"] = df["prem"] - df["prem"].rolling(60, min_periods=5).mean()
    # alt / cross-book features on the recent per-second window (only for the last_n bars)
    alts = {}
    for a in altfeat.ALTS + (altfeat.BOOKS if wide else ()):
        x = Es if a == LEAD_KEY else src.spot_seconds(a, s0_ms, SEC_WIN)
        alts[a] = _arrays(x, ["c", "v", "bv", "n"])
    Srec = _arrays(S.iloc[nS - SEC_WIN:].reset_index(drop=True), ["c", "v"])
    if hasattr(src, "coinbase_minutes"):
        try:
            cb = src.coinbase_minutes(h0_ms, HIST_MIN)
        except Exception:  # noqa: BLE001
            cb = None
        if cb is not None and len(cb):
            from ..features import coinbase
            CBF = coinbase.features(df[["ts", "s_close", "sigma", "s_vol60"]], cb).iloc[-last_n:]
        else:
            CBF = None
    else:
        CBF = None
    endr = SEC_WIN - 1 - 60 * np.arange(last_n)[::-1]
    A = altfeat.alt_from_arrays(Srec, {k: alts[k] for k in altfeat.ALTS}, {k: alts[k] for k in altfeat.BOOKS if k in alts}, endr,
                                df["ts"].values[-last_n:])
    A = A.drop(columns="ts").reset_index(drop=True)
    # wide swarm (17 alts) + ETH perpetual
    alts_all = dict(alts)
    for a2 in (altfeat.ALTS2 if wide else ()):
        alts_all[a2] = _arrays(src.spot_seconds(a2, s0_ms, SEC_WIN), ["c", "v", "bv"])
    if use_futures:
        Frec = _arrays(Fs, ["c", "v"])
        xperp = _arrays(src.fut_seconds(s0_ms, SEC_WIN, LEAD_PERP_SYMBOL), ["c", "v", "bv"])
    else:
        Frec, xperp = None, None
    A2 = altfeat.alt2_from_arrays(Srec, Frec, {k: alts_all[k] for k in altfeat.ALTS + altfeat.ALTS2 if k in alts_all}, xperp,
                                  endr, df["ts"].values[-last_n:])
    if not wide:
        A2 = A2.drop(columns=[c for c in A2.columns if c.startswith("swarm2_")])
    A = pd.concat([A, A2.drop(columns="ts").reset_index(drop=True)], axis=1)
    if CBF is not None:
        A = pd.concat([A, CBF.reset_index(drop=True)], axis=1)
    return df.reset_index(drop=True), A


def _prefetch(src, h0_ms, nS, s0_ms, use_futures, wide=True):
    """Issue all network requests of one step concurrently; build_features then hits the cache."""
    from concurrent.futures import ThreadPoolExecutor

    calls = [lambda: src.spot_seconds(TARGET_KEY, h0_ms, nS), lambda: src.minutes("lead", h0_ms, HIST_MIN),
             lambda: src.spot_seconds(LEAD_KEY, s0_ms, SEC_WIN), lambda: src.coinbase_minutes(h0_ms, HIST_MIN)]
    calls += [(lambda a=a: src.spot_seconds(a, s0_ms, SEC_WIN))
              for a in altfeat.ALTS + ((altfeat.BOOKS + altfeat.ALTS2) if wide else ()) if a != LEAD_KEY]
    if use_futures:
        calls += [lambda: src.minutes("perp", h0_ms, HIST_MIN), lambda: src.fut_seconds(s0_ms, SEC_WIN),
                  lambda: src.fut_seconds(s0_ms, SEC_WIN, LEAD_PERP_SYMBOL), lambda: src.premium(h0_ms - 60000 * 600, HIST_MIN + 600)]
    with ThreadPoolExecutor(12) as ex:
        for f in [ex.submit(c) for c in calls]:
            try:
                f.result()
            except Exception:  # noqa: BLE001 - surfaced again by the sequential call
                pass


def model_inputs(df_hist: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
    """Columns identical to training: model_matrix + extra_features + alt features, for the last
    len(A) bars (rolling statistics use the full minute history in df_hist)."""
    n = len(A)
    X = features.model_matrix(df_hist).iloc[-n:]
    Xe = features.extra_features(df_hist).iloc[-n:]
    out = pd.concat([X.reset_index(drop=True), Xe.reset_index(drop=True), A.astype("float32")], axis=1)
    out.index = df_hist.index[-n:]
    return out


# =============================================================================================
class CachedSource:
    """Memoizes per-second and per-minute data so that each live step only downloads new data.
    The newest `fresh` seconds/minutes are always re-fetched (late trades)."""

    def __init__(self, src, fresh_s: int = 3):
        self.src = src
        self.fresh_s = fresh_s
        self.cache: dict = {}

    def _get(self, key, unit_ms, fetch, t0_ms, n):
        memo = self.cache.get(("memo", key, t0_ms, n))
        if memo is not None and time.time() - memo[0] < 20:
            return memo[1]
        res = self._get_inner(key, unit_ms, fetch, t0_ms, n)
        now = time.time()
        for k in [k for k, v in self.cache.items() if k[0] == "memo" and now - v[0] > 120]:
            del self.cache[k]
        self.cache[("memo", key, t0_ms, n)] = (now, res)
        return res

    def _get_inner(self, key, unit_ms, fetch, t0_ms, n):
        t1_ms = t0_ms + n * unit_ms
        c = self.cache.get(key)
        if c is not None:
            c0, df = c
            c1 = c0 + len(df) * unit_ms
            valid_end = c1 - (self.fresh_s * 1000 if unit_ms == 1000 else 2 * unit_ms)
            if c0 <= t0_ms < valid_end:
                k_keep = (valid_end - c0) // unit_ms
                new_n = (t1_ms - valid_end) // unit_ms
                if new_n > 0:
                    new = fetch(valid_end, int(new_n))
                    df = pd.concat([df.iloc[:k_keep], new], ignore_index=True)
                else:
                    df = df.iloc[:(t1_ms - c0) // unit_ms]
                # trim to what we will ever need again
                drop = max(0, (t0_ms - c0) // unit_ms)
                df = df.iloc[drop:].reset_index(drop=True)
                self.cache[key] = (t0_ms, df)
                return df.iloc[:n].reset_index(drop=True)
        df = fetch(t0_ms, n)
        self.cache[key] = (t0_ms, df)
        return df

    def spot_seconds(self, symbol_key, t0_ms, n):
        return self._get(("s", symbol_key), 1000, lambda a, k: self.src.spot_seconds(symbol_key, a, k), t0_ms, n)

    def fut_seconds(self, t0_ms, n, symbol=PERP_SYMBOL):
        return self._get(("f", symbol), 1000, lambda a, k: self.src.fut_seconds(a, k, symbol), t0_ms, n)

    def minutes(self, venue, t0_ms, n_min):
        def fetch(a, k):
            m = self.src.minutes(venue, a, k)
            full = pd.DataFrame({"open_ms": a + 60000 * np.arange(k)})
            return full.merge(m, on="open_ms", how="left")
        return self._get(("m", venue), 60000, fetch, t0_ms, n_min)

    def premium(self, t0_ms, n_min):
        def fetch(a, k):
            m = self.src.premium(a, k)
            full = pd.DataFrame({"open_ms": a + 60000 * np.arange(k)})
            return full.merge(m, on="open_ms", how="left")
        return self._get(("p",), 60000, fetch, t0_ms, n_min)

    def coinbase_minutes(self, t0_ms, n_min):
        memo = self.cache.get(("memo", "cb", t0_ms, n_min))
        if memo is not None and time.time() - memo[0] < 20:
            return memo[1]
        out = self._coinbase(t0_ms, n_min)
        self.cache[("memo", "cb", t0_ms, n_min)] = (time.time(), out)
        return out

    def _coinbase(self, t0_ms, n_min):
        """Candles are fetched only once they are >= CB_FRESH_MIN minutes old (final); the recent
        minutes come from an incrementally maintained cache of raw trades."""
        R = self.src
        t0 = pd.Timestamp(t0_ms, unit="ms")
        stop = t0 + pd.Timedelta(minutes=n_min)
        cut = stop - pd.Timedelta(minutes=R.CB_FRESH_MIN)
        cand = self.cache.get("cb_candles")
        final_until = self.cache.get("cb_final_until")
        if cand is None or final_until is None or final_until < t0:
            cand = R.coinbase_candles(t0, cut)
        elif cut > final_until:
            cand = pd.concat([cand, R.coinbase_candles(final_until, cut)], ignore_index=True)
        cand = cand[(cand["ts"] >= t0) & (cand["ts"] < cut)].drop_duplicates("ts", keep="last")
        self.cache["cb_candles"], self.cache["cb_final_until"] = cand, cut
        tr = self.cache.get("cb_trades")
        if tr is None or len(tr) == 0 or tr["t"].min() > cut:
            tr = R.coinbase_trades(cut)
        else:
            tr = pd.concat([tr, R.coinbase_trades(cut, stop_id=int(tr["trade_id"].max()))], ignore_index=True)
        tr = tr[tr["t"] >= cut - pd.Timedelta(minutes=2)].drop_duplicates("trade_id")
        # completeness barrier: the last minute is closed only once a trade stamped >= stop is visible
        for _ in range(8):
            if len(tr) and tr["t"].max() >= stop:
                break
            time.sleep(0.25)
            new = R.coinbase_trades(cut, stop_id=int(tr["trade_id"].max()) if len(tr) else None)
            tr = pd.concat([tr, new], ignore_index=True).drop_duplicates("trade_id")
        self.cache["cb_trades"] = tr
        recent = R.trades_to_minutes(tr, cut, stop)
        return pd.concat([cand, recent], ignore_index=True).sort_values("ts").reset_index(drop=True)
