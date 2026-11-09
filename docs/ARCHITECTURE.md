# Architecture

This is a technical reference. For results see `README.md`; for why things are the way they are,
see `docs/FINDINGS.md`.

The machinery is shared with `btc-predict`. Everything asset-specific lives in the **asset
profile** in `ethpredict/config.py`:
* symbols: target ETHUSDT spot/perp, "other major" BTCUSDT spot/perp, Coinbase product ETH-USD;
* per-second data directory names;
* `LARGE_QTY = 25` ETH;
* the cross-asset universe.

## 1. Data flow

```
 public endpoints                     data/sec/<venue>/<day>.parquet          data/features/
 ───────────────                     (one row per second, 86 400/day)        (one row per minute)
 data.binance.vision ─┐                                                       panel.parquet  (+ minute targets)
   ETH spot 1s klines │  ethpredict.data.binance   ─────────────────────►     X.parquet      (model matrix)
   ETH perp aggTrades │  ethpredict.data.coinbase    ethpredict.features.*    Xalt / Xalt2   (cross-asset)
   BTC spot/perp, alts│                                                       Xcb            (Coinbase)
 Coinbase ETH-USD ────┘                                                       Ysub           (10 s / 15 s targets)
                                                                                   │
                       scripts/pipeline/train_final.py  ◄────────────────────────┘
                         walk-forward on the HOLDOUT → reports/holdout/*.json, reports/preds/*.parquet
                         production LightGBM (all data) → models/*.txt(.gz) + models/meta.json
                       scripts/pipeline/add_regime_calibration.py → bands / hit tables into meta.json
                                                                                   │
 live REST APIs ──► ethpredict.live.sources (same feature code) ──► ethpredict.live.predictor ──► CLI
```

`data/sec` may be a symlink to a store shared with btc-predict; the directory names are the
same. Per-second directories used:
* `eth`: ETH spot;
* `ethum`: ETH perp;
* `spot` / `um`: BTC spot and perp;
* `sol`, `xrp`, `doge`, `bnb`: the swarm;
* `eth_premium`;
* `coinbase/coinbase_ETH-USD_1m.parquet`.

The research-only directories are `ethusdc`, `ethbtc`, the 12 extra alts and `ethum1m`.

## 2. Time conventions (the most important thing to get right)

* **Row `ts`** is the bar *open* time. The **forecast instant** is `T = ts + 60 s` (the bar close).
* **`P(T)`** is the last trade price before `T`, i.e. the close of second `T−1 s`.
* **Features** may use only seconds `≤ T−1 s`. In the code, `end` is the index of the bar's last
  second.
* **Minute targets:** `y_{v}{h} = log P(T+h) − log P(T)`, with `v` ∈ {`s` = Binance spot
  ETHUSDT, `f` = USD-M perp ETHUSDT} and `h` = 1…5.
* **Second targets:** `y_{v}10s`, `y_{v}15s` (`features/subminute.py`).
* **Delivery-delay variants** (honest live metrics):
  * minutes: `y_{v}{h}_d{D} = log P(T+h) − log P(T+D)`
  * seconds: `y_{v}{H}s_d{D} = log P(T+D+H) − log P(T+D)`
* **Model targets are scale-free:** `z = y / (sigma · sqrt(h_minutes))`. Here `sigma` is the
  per-minute volatility, an EWM (span 30 minutes) of the per-minute realized variance from
  1-second returns.
* **Timestamp quirk:** Binance spot 1 s files use **microsecond** timestamps since 2025 (handled
  in `_to_ms`).

## 3. Package map

| module | role |
|---|---|
| `config.py` | every path and constant, plus the asset profile |
| `data/binance.py` | bulk daily files → per-second bars. Jobs: `eth`, `ethum` (perp aggTrades), `spot`, `um` (BTC), alts, `eth_premium`; research: `ethusdc`, `ethbtc`, `ethum1m`. `trades_to_1s()` aggregates ticks; "large" trades are ≥ `LARGE_QTY`. |
| `data/coinbase.py` | Coinbase ETH-USD 1 m candles (REST) |
| `features/seconds.py` | `features_from_arrays()`: multi-window (1–600 s) returns, signed flow, volume, trade count and realized variance for spot & perp; perp–spot **basis** deviation/changes; Lévy areas; BTC spot (`e_*`); forward prices for targets |
| `features/panel.py` | stitches days, adds **minute-level** features (`add_minute_features`: vol state, multi-minute returns, flow, Kyle λ, ranges, VWAP, basis, BTC beta `x_beta`, calendar), the perp premium index, and minute targets |
| `features/matrix.py` | `model_matrix()` makes everything scale-free; `extra_features()` adds reversal kernels and quarter-hour interactions; `target_scale()` |
| `features/crossasset.py` | 5-coin swarm incl. BTC (`alt_*`), other books (`book_*`, research) — `alt_from_arrays`; 17-coin swarm (`swarm2_*`, research) + BTC perp (`xp_*`, `x_basis_dev*`) — `alt2_from_arrays` |
| `features/coinbase.py` | Coinbase premium features (`cb_*`) |
| `features/subminute.py` | 10 s / 15 s targets |
| `modeling/gbm.py` | `lgbm_factory()` (Huber, 400 rounds, lr 0.03, 31 leaves, ≥ 2000 rows/leaf, feature_fraction 0.5, bagging 0.7, L2 10, target clipped ±6), `ridge_factory`, `Standardizer` |
| `modeling/online.py` | `PriorRLS`: discounted ridge with a persistent prior (production ensemble); `OnlineRLS`: plain RLS |
| `modeling/calibrate.py` | causal logistic P(up), log-loss/reliability, adaptive conformal (research) |
| `modeling/evaluate.py` | `make_folds` (expanding walk-forward, purge, month blocks), `metrics`, `summarize` (incl. delay variants), `monthly_ic`, `paired_block_bootstrap` |
| `live/sources.py` | `RestSource` (official endpoints), `ReplaySource` (historical files, same interface), `CachedSource` (incremental, concurrent prefetch, Coinbase trades cache + completeness barrier). `build_features()` and `model_inputs()` run the **same** feature functions as training. `wide=False` skips the inputs no production model uses. |
| `live/predictor.py` | loads `models/meta.json` and boosters (`.txt` or `.txt.gz`). `predict()` → P(up), E[r], bands, confidence, regime. `update()` drives online learning; state is persisted in `logs/`. `needs_wide` tells the live loop whether research-only inputs are needed. |
| `live/cli.py` | the minute loop: fetch at T+2.0 s exchange time (clock-offset corrected), forecast, log, learn |

## 4. Feature families → feature sets

The feature lists live in `configs/feature_sets/*.txt`. They are generated from feature
families by `scripts/research/make_family_sets.py`; see the README there.

| set | used by | contents |
|---|---|---|
| `core` / `core_fast` | full model, core part | ETH spot + perp microstructure, basis, perp premium index, minute state, calendar, BTC spot |
| `sat` / `sat_fast` | full model, satellite part | core + Coinbase ETH premium + BTC perp + 5-coin swarm |
| `spotcore`, `spotsat` (+`_fast`) | spot-only model | the same, without any futures-derived column |

The `_fast` sets drop slow "regime" features (24 h returns, hour-of-day, etc.) and are used for
h ≥ 2 minutes (`train_final.py --fast_from 2`). That rule was re-derived for ETH: see FINDINGS §4.

## 5. Models and ensemble (per variant × target)

* There are **two variants**: `full` (targets f/s × 10s,15s,1–5m) and `spot` (s × 10s,15s,1–5m).
* **Core and satellite LightGBM**, each predicting the z-target. They are trained on all data for
  production and walk-forward for evaluation.
* **Ensemble** `z = w · [z_core, z_sat]` (no intercept), using `PriorRLS` with prior `[0, 1]`
  ("trust the satellite") and a half-life of 3 days.
  * h = 1 m and 10 s/15 s: κ = 10, adaptive. It learns live from every realized outcome and
    falls back to the core if a cross-venue relationship breaks.
  * h ≥ 2 m: κ = 1e12, frozen on the satellite, because overlapping outcomes make online weights
    noise.

## 6. Calibration and outputs (`meta.json` per target)

* **P(up):** `sigmoid(a + b·z)` (`calib`).
* **E[r]:** `beta · z · sigma · sqrt(h)`.
* **Bands:** `E[r] + q · sigma · sqrt(h)`, where `q` comes from `band_table`. That table is
  indexed by `vrel = log(sigma / 1-day median sigma)` and multiplied by two factors:
  * `et_hour_band_factor[New-York hour]`;
  * `vr_band_factor[variance-ratio tercile]`.
* **Confidence:** the percentile of |z| among holdout forecasts.
* **hist hit:** the realized hit rate for that (|z| decile × volatility regime).

Full schema: `models/README.md`.

## 7. Evaluation protocol

* **DEV tuning months:** 2025-11, 2026-01, 2026-03. **DEV confirmation months** (never tuned
  on): 2025-12, 2026-02, 2026-04.
* **HOLDOUT:** 2026-05 → 2026-10, frozen, and the source of the headline numbers.
* **Walk-forward:** expanding window; training rows purged h + 10 min before each test block;
  the final run retrains every 2 months.
* **Decisions** use a paired day-block bootstrap of the IC difference
  (`scripts/research/compare_preds.py`).
* **Metrics:** IC, rank IC, hit rate (non-zero moves), AUC, top-k% confidence hit, log-loss
  skill, delivery-delay variants, monthly IC.

## 8. Live loop timing (`live/cli.py`)

1. At start-up it backfills 2600 minutes of history and measures the exchange clock offset.
2. Every minute, at **T + 2.0 s exchange time**, it starts. Binance 1 s klines are final from
   ~1.5 s; fetching earlier causes training/serving skew.
3. It issues concurrent requests:
   * ETH spot, BTC spot and 4 swarm-coin 1 s klines;
   * perp aggTrades and klines for ETH and BTC, and the ETH premium index (full variant);
   * Coinbase ETH-USD trades (last 15 min, plus a completeness barrier) and candles (older
     minutes).
4. It builds features, predicts, prints, and appends to `logs/predictions.jsonl` and
   `logs/live_features_<variant>.csv`.
5. It updates the online layer with matured forecasts and saves `logs/rls_state_<variant>.json`.

## 9. Invariants (break these and results silently become wrong)

1. **No look-ahead.** Features use seconds ≤ T−1 s only, and targets live in `y_*` / `px_*`
   columns that are never features.
2. **One feature code path.** Any feature change goes into `features/*` *and* is automatically
   used by `live/sources.py`. Run `make parity` (feature + score parity) after every change.
3. **Booster column order = `meta.json["features"]`.** The predictor refuses to run when a column
   is missing or the count does not match.
4. **Never tune on the HOLDOUT months.** Use the DEV months, then confirm on the confirmation
   months.
5. **New live data sources need a skew test** (`make skew`): exchanges revise and publish late.
6. **`LARGE_QTY` must equal the value the stored `ethum` seconds were built with** (25 ETH).
7. **Binance USD-M futures REST is geo-blocked** in some regions. Use `--variant spot` there.
