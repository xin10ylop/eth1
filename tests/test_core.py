"""Fast unit tests for the core math (no market data needed).  Run: python -m pytest -q tests"""
import numpy as np
import pandas as pd

from ethpredict.data import binance as download
from ethpredict.features import seconds as secfeat
from ethpredict.modeling import online


def test_trades_to_1s_aggregation():
    # three trades in second 0 (two buys, one sell), one trade in second 2
    t = np.array([100, 500, 900, 2100])
    px = np.array([10.0, 11.0, 9.0, 12.0])
    L = download.LARGE_QTY
    q = np.array([1.0, 2.0, 0.5, 1.2 * L])
    buyer_maker = np.array([False, False, True, False])     # aggressor buys, buys, sells, buys
    df = download.trades_to_1s(t, px, q, buyer_maker, n=3)
    assert df.loc[0, "o"] == 10.0 and df.loc[0, "c"] == 9.0
    assert df.loc[0, "h"] == 11.0 and df.loc[0, "l"] == 9.0
    assert df.loc[0, "v"] == 3.5 and df.loc[0, "bv"] == 3.0 and df.loc[0, "n"] == 3
    assert df.loc[0, "lbv"] == 0                                  # small trades are not "large"
    assert np.isnan(df.loc[1, "c"]) and df.loc[1, "v"] == 0
    assert df.loc[2, "c"] == 12.0 and df.loc[2, "lbv"] == 1.2 * L  # >= LARGE_QTY ETH is a "large" buy


def test_levy_area_matches_direct_sum():
    rng = np.random.default_rng(0)
    X = np.cumsum(rng.normal(size=500)); Y = np.cumsum(rng.normal(size=500))
    end = np.array([200, 350, 499]); w = 60
    got = secfeat._levy(X, Y, end, w)
    for e, g in zip(end, got):
        a = e - w
        s = sum((X[i - 1] - X[a]) * (Y[i] - Y[i - 1]) - (Y[i - 1] - Y[a]) * (X[i] - X[i - 1]) for i in range(a + 1, e + 1))
        assert abs(0.5 * s - g) < 1e-9


def test_levy_area_sign_detects_leader():
    # Y copies X with a 3-step lag -> X leads -> positive area for (X, Y)
    rng = np.random.default_rng(1)
    X = np.cumsum(rng.normal(size=2000))
    Y = np.r_[np.zeros(3), X[:-3]]
    a = secfeat._levy(X, Y, np.array([1999]), 1500)[0]
    assert a > 0


def test_window_sums_use_only_past():
    x = np.arange(100, dtype=float)
    cs = secfeat._cs(x)
    end = np.array([50])
    assert secfeat._wsum(cs, end, 5)[0] == sum(range(46, 51))   # indices 46..50 inclusive, nothing after 50


def test_prior_rls_stays_on_prior_without_evidence_and_moves_with_it():
    rng = np.random.default_rng(2)
    n = 20000
    core = rng.normal(size=n); sat = rng.normal(size=n)
    # truth: only the core is informative -> weights must move from prior [0,1] toward the core
    y = 0.3 * core + rng.normal(size=n)
    Z = np.column_stack([core, sat])
    _, m = online.prior_stack(Z, y, horizon=1, half_life=4320, kappa=10, prior=[0.0, 1.0], return_model=True)
    assert m.w[0] > 0.2 and m.w[1] < 0.3
    # infinite prior strength keeps the prior
    _, m2 = online.prior_stack(Z, y, horizon=1, half_life=4320, kappa=1e12, prior=[0.0, 1.0], return_model=True)
    assert np.allclose(m2.w, [0.0, 1.0], atol=1e-6)


def test_prior_rls_state_roundtrip():
    m = online.PriorRLS(2, prior=[0.0, 1.0])
    for i in range(100):
        m.update([1.0, 0.5], 0.1 * i)
    m2 = online.PriorRLS.from_state(m.state())
    assert np.allclose(m.w, m2.w) and m2.predict([1.0, 1.0]) == m.predict([1.0, 1.0])


def test_minute_embedding_preserves_minute_aggregates():
    from ethpredict.live import sources as live
    t0 = 1_700_000_000_000
    m = pd.DataFrame({"open_ms": t0 + 60000 * np.arange(3), "c": [1.0, 2.0, 3.0], "h": [1.5, 2.5, 3.5],
                      "l": [0.5, 1.5, 2.5], "v": [10.0, 0.0, 5.0], "qv": [10.0, 0.0, 15.0], "n": [3, 0, 2], "bv": [6.0, 0.0, 1.0]})
    s = live.minute_to_seconds(m, t0, 180)
    g = np.arange(180) // 60
    assert np.allclose(s["v"].groupby(g).sum().values, m["v"].values)
    assert np.allclose(s["bv"].groupby(g).sum().values, m["bv"].values)
    assert np.allclose(pd.Series(s["c"]).groupby(g).last().values, m["c"].values)
