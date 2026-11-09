"""Load trained artifacts and turn feature rows into calibrated forecasts.

Per target: z_core, z_sat (LightGBM) -> z = RLS([1, z_core, z_sat]) -> P(up), E[r], bands.
The RLS layer keeps learning live (update()) and its state is persisted between runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LOGS, MODELS
from ..modeling.online import OnlineRLS, PriorRLS


def _load_booster(path: Path):
    """Load a LightGBM text model, falling back to the gzip-compressed copy shipped in the repo."""
    import gzip

    import lightgbm as lgb

    if path.exists():
        return lgb.Booster(model_file=str(path))
    gz = path.with_name(path.name + ".gz")
    with gzip.open(gz, "rt") as fh:
        return lgb.Booster(model_str=fh.read())


class Predictor:
    def __init__(self, variant: str = "full", model_dir: Path = MODELS, state_file: Path | None = None):
        self.dir = Path(model_dir)
        meta = json.loads((self.dir / "meta.json").read_text())
        if variant not in meta:
            raise ValueError(f"variant {variant!r} not trained; available: {list(meta)}")
        self.meta = meta[variant]
        self.variant = variant
        self.boosters = {k: {part: _load_booster(self.dir / f) for part, f in m["files"].items()}
                         for k, m in self.meta["targets"].items()}
        self.state_file = state_file or (LOGS / f"rls_state_{variant}.json")
        saved = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        self.rls = {}
        for k, m in self.meta["targets"].items():
            st = saved.get(k, m["rls_state"])
            self.rls[k] = PriorRLS.from_state(st) if st.get("type") == "prior" else OnlineRLS.from_state(st)

    @property
    def uses_futures(self) -> bool:
        return self.meta["uses_futures"]

    @property
    def needs_wide(self) -> bool:
        """True if any booster uses the 17-coin swarm or another book (see sources.build_features)."""
        return any(c.startswith(("swarm2_", "book_")) for t in self.meta["targets"].values()
                   for cols in t.get("features", {}).values() for c in cols)

    def raw(self, X: pd.DataFrame) -> dict:
        out = {}
        for key, parts in self.boosters.items():
            out[key] = {}
            for p, b in parts.items():
                names = self.meta["targets"][key].get("features", {}).get(p) or b.feature_name()
                if b.num_feature() != len(names):
                    raise RuntimeError(f"{key}/{p}: model expects {b.num_feature()} features, meta lists {len(names)}")
                missing = [c for c in names if c not in X.columns]
                if missing:
                    raise RuntimeError(f"{key}/{p}: {len(missing)} features missing from live inputs, e.g. {missing[:5]}")
                out[key][p] = b.predict(X[names].values.astype(np.float64))
        return out

    def regime(self, vrel: float) -> tuple[int, str]:
        """Volatility regime from vrel = log(sigma / 1-day median sigma)."""
        r = self.meta.get("regime")
        if r is None or not np.isfinite(vrel):
            return 1, "normal"
        g = int(np.digitize(vrel, r["regime_edges"]))
        return g, r["regime_names"][g]

    def predict(self, X: pd.DataFrame, sigma: np.ndarray, vrel: np.ndarray | None = None,
                T: pd.Timestamp | None = None) -> pd.DataFrame:
        """X: feature rows; sigma: per-minute volatility (log-return units); vrel: log(sigma / 1-day
        median sigma) -> regime-conditional bands and hit rates (falls back to static if absent)."""
        raw = self.raw(X)
        vrel = np.full(len(X), np.nan) if vrel is None else np.asarray(vrel, float)
        reg = self.meta.get("regime")
        rows = []
        for key, m in self.meta["targets"].items():
            zc, zs = raw[key]["core"], raw[key]["sat"]
            z = np.array([self.rls[key].predict(self._x(key, a, b)) for a, b in zip(zc, zs)])
            h = m["h"]
            scale = sigma * np.sqrt(h)
            a, bb = m["calib"]
            p_up = 1 / (1 + np.exp(-(a + bb * z)))
            exp_ret = m["beta"] * z * scale
            conf = [float(np.mean(abs(zi) >= np.asarray(m["absz_quantiles"])) * 100) for zi in z]
            for i in range(len(X)):
                g, gname = self.regime(vrel[i])
                if reg is not None and "band_table" in m and np.isfinite(vrel[i]):
                    q5 = m["band_table"][int(np.digitize(vrel[i], reg["band_edges"]))]
                    q = dict(zip(("0.025", "0.1", "0.5", "0.9", "0.975"), q5))
                else:
                    q = dict(m["resid_q"])
                k = 1.0
                if T is not None and "et_hour_band_factor" in m:
                    et_h = pd.Timestamp(T).tz_localize("UTC").tz_convert("America/New_York").hour
                    k *= m["et_hour_band_factor"][et_h]
                if "vr_band_factor" in m and "vr_1s_vs_1m" in X.columns and np.isfinite(X["vr_1s_vs_1m"].iloc[i]):
                    k *= m["vr_band_factor"][int(np.digitize(X["vr_1s_vs_1m"].iloc[i], m["vr_edges"]))]
                q = {kk: (vv if kk == "0.5" else vv * k) for kk, vv in q.items()}
                if "hit_table" in m:
                    d = int(np.digitize(abs(z[i]), m["absz_deciles"]))
                    hh = m["hit_table"][d][g]
                else:
                    dec = m.get("hit_by_conf_decile") or [np.nan] * 10
                    hh = dec[min(int(conf[i] // 10), len(dec) - 1)]
                label = m.get("label") or (f"{round(h * 60)}s" if h < 1 else f"{int(h)}m")
                label = label if label.endswith(("s", "m")) else f"{label}m"
                rows.append({"target": key, "venue": m["venue"], "h": h, "label": label, "z": z[i], "z_core": zc[i], "z_sat": zs[i],
                             "p_up": p_up[i], "exp_ret_bp": 1e4 * exp_ret[i],
                             "lo80_bp": 1e4 * (exp_ret[i] + q["0.1"] * scale[i]), "hi80_bp": 1e4 * (exp_ret[i] + q["0.9"] * scale[i]),
                             "lo95_bp": 1e4 * (exp_ret[i] + q["0.025"] * scale[i]), "hi95_bp": 1e4 * (exp_ret[i] + q["0.975"] * scale[i]),
                             "confidence_pct": conf[i], "regime": gname, "hist_hit_at_conf": hh})
        return pd.DataFrame(rows)

    def _x(self, key, z_core, z_sat):
        # layouts: [core, sat] (current) or [1, core, sat] (older artifacts)
        return [z_core, z_sat] if len(self.rls[key].w) == 2 else [1.0, z_core, z_sat]

    def update(self, key: str, z_core: float, z_sat: float, realized_z: float):
        self.rls[key].update(self._x(key, z_core, z_sat), realized_z)

    def save_state(self):
        self.state_file.parent.mkdir(exist_ok=True)
        self.state_file.write_text(json.dumps({k: r.state() for k, r in self.rls.items()}))
