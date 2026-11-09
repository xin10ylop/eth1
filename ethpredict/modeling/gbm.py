"""Model zoo. Every model is a function fit_predict(Xtr, ytr, Xte, wtr) -> np.ndarray."""
from __future__ import annotations

import numpy as np
import pandas as pd

CLIP_Z = 6.0


def lgbm_factory(params: dict | None = None, rounds: int = 600, clip: float = CLIP_Z, objective: str = "huber",
                 valid_frac: float = 0.0, feature_subset: list[str] | None = None):
    import lightgbm as lgb

    base = dict(objective=objective, alpha=1.5 if objective == "huber" else None, learning_rate=0.03, num_leaves=31,
                min_data_in_leaf=2000, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1,
                lambda_l2=10.0, max_bin=127, verbose=-1, num_threads=4, seed=7)
    if params:
        base.update(params)
    base = {k: v for k, v in base.items() if v is not None}

    def fp(Xtr, ytr, Xte, wtr=None):
        if feature_subset is not None:
            Xtr, Xte = Xtr[feature_subset], Xte[feature_subset]
        y = np.clip(ytr.values, -clip, clip)
        ds = lgb.Dataset(Xtr.values, y, weight=wtr, free_raw_data=True, feature_name=[str(c) for c in Xtr.columns])
        model = lgb.train(base, ds, num_boost_round=rounds)
        fp.last_model = model
        fp.last_features = list(Xtr.columns)
        return model.predict(Xte.values, num_threads=4)

    return fp


class Standardizer:
    def fit(self, X: pd.DataFrame):
        self.mu = X.median()
        q = X.quantile([0.01, 0.99])
        self.lo, self.hi = q.iloc[0], q.iloc[1]
        Xc = X.clip(self.lo, self.hi, axis=1)
        self.sd = Xc.std().replace(0, 1).fillna(1)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = ((X.clip(self.lo, self.hi, axis=1) - self.mu) / self.sd).fillna(0.0)
        return Z.values.astype(np.float64)


def ridge_factory(alpha: float = 1e3, clip: float = CLIP_Z, feature_subset: list[str] | None = None):
    def fp(Xtr, ytr, Xte, wtr=None):
        if feature_subset is not None:
            Xtr, Xte = Xtr[feature_subset], Xte[feature_subset]
        st = Standardizer().fit(Xtr)
        A = st.transform(Xtr)
        y = np.clip(ytr.values, -clip, clip)
        w = np.ones(len(y)) if wtr is None else wtr
        Aw = A * w[:, None]
        G = A.T @ Aw + alpha * np.eye(A.shape[1])
        b = np.linalg.solve(G, Aw.T @ y)
        fp.coef = pd.Series(b, index=Xtr.columns)
        return st.transform(Xte) @ b

    return fp


def zero_model(Xtr, ytr, Xte, wtr=None):
    return np.zeros(len(Xte))


def single_feature(col: str, sign: float = 1.0):
    def fp(Xtr, ytr, Xte, wtr=None):
        x = Xtr[col].values; y = np.clip(ytr.values, -CLIP_Z, CLIP_Z)
        m = np.isfinite(x)
        b = np.polyfit(x[m], y[m], 1)
        return np.polyval(b, np.nan_to_num(Xte[col].values))
    return fp
