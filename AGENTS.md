# Agent guide: eth-predict, from A to Z

You are working on **eth-predict** (package `ethpredict`). It forecasts Ethereum's price change
over the next **10 s, 15 s, 1, 2, 3, 4 and 5 minutes**, every minute, for Binance spot ETH/USDT
and the Binance USD-M perpetual. Your goal is to understand it fully before changing anything,
and to keep it honest: no look-ahead, no tuning on the holdout, and every claim backed by an
out-of-sample number.

It is a sibling of `btc-predict` and shares its machinery. **The research was redone for ETH, not
copied.** Several BTC conclusions do not hold for ETH (see §3, concept 11).

## 1. Read these first (in order, ~15 minutes)
1. `README.md`: what it does, the headline results, how to run it.
2. `docs/ARCHITECTURE.md`: data flow, time conventions, modules, models, live loop, invariants.
3. `docs/FINDINGS.md`: the research log. It covers ETH vs BTC mechanisms, what survived for ETH,
   what was rejected and why, regime change, and training/serving skew.
4. `models/README.md`, `configs/feature_sets/README.md`, `reports/README.md`: artifacts.

## 2. Repo map
```
ethpredict/            the package, one subpackage per pipeline stage
  config.py            every path & constant + the ASSET PROFILE (symbols, data dirs, LARGE_QTY)
  data/                1. download: binance.py (bulk files -> per-second bars), coinbase.py
  features/            2. features & targets: seconds.py, panel.py, matrix.py, crossasset.py,
                          coinbase.py, subminute.py   (same code runs in training AND live)
  modeling/            3. gbm.py (LightGBM), online.py (PriorRLS ensemble), calibrate.py, evaluate.py
  live/                4. sources.py (REST/replay/cache), predictor.py, cli.py  -> python -m ethpredict.live
configs/feature_sets/  which columns each model uses (+ research/ family-defined sets)
models/                production boosters (*.txt.gz) + meta.json (calibration, features, online state)
scripts/pipeline/      build_features, train_final (holdout + production), add_regime_calibration, monthly_refresh
scripts/live/          score_live_log, check_live_parity, live_skew_now, check_live_skew
scripts/reports/       results_table, make_report (charts), edge_vs_costs
scripts/research/      run_model.py (one walk-forward experiment), compare_preds.py (paired bootstrap),
                       make_family_sets.py, slices.py, mechanisms/ (ETH-vs-BTC hypothesis tests)
reports/               holdout/ (headline numbers), confirm/, figures/, dev/, research/, live/
tests/                 unit tests for the core math
data/, logs/           local only (not in git): per-second data, feature matrices, live logs
```

## 3. Concepts you must know
1. **Forecast instant.** Row `ts` = bar open; forecasts are made at `T = ts + 60 s` using only
   data before `T`. `P(T)` = last trade before T.
2. **Venues.** `s` = Binance spot ETHUSDT, `f` = USD-M perp. The perp leads; spot catches up
   within seconds. The ETH perp is a price-discovery venue, which is why it is much harder to
   forecast than spot.
3. **Targets.** Log return from T to T + horizon. Models learn `z = return / (sigma·sqrt(h))`,
   which is scale-free.
4. **Two variants.**
   * `full`: needs Binance futures.
   * `spot`: no futures data; use it where futures are geo-blocked.
5. **Core + satellite.**
   * Core: LightGBM on ETH spot + perp microstructure, perp premium, BTC spot, minute state.
   * Satellite: core + Coinbase ETH premium + BTC perp + 5-coin swarm. The ETH/USDC book and the
     17-coin swarm were tested and **left out** (no gain for ETH).
6. **The "other major".** In this repo BTC plays the role ETH played in btc-predict: `e_*`,
   `be_*`, `E*`, `BE*`, `x_beta`, `x_resid5` = BTC spot; `xp_*`, `x_basis_dev*` = BTC perp.
7. **Online ensemble.** `PriorRLS` blends core and satellite, learns from live outcomes at
   h ≤ 1 m, and falls back to the core if a cross-venue relationship breaks.
8. **Calibration.**
   * P(up) is logistic in z.
   * Bands are residual quantiles conditioned on the volatility regime, New-York hour and
     variance ratio.
9. **Evaluation.**
   * DEV months are for decisions, confirmation months for checking them.
   * The **HOLDOUT (2026-05 → 2026-10) is frozen** and is the source of the headline numbers.
   * Changes are accepted only with a paired day-block bootstrap
     (`scripts/research/compare_preds.py`).
10. **Latency and skew.**
    * The edge decays within seconds. The CLI fetches at T + 2.0 s (exchange data are final from
      ~1.5 s).
    * Live features must equal features recomputed later from final data. Coinbase revises
      recent candles, so recent minutes are rebuilt from trades.
    * Check with `make parity` and `make skew`.
11. **ETH is not BTC.**
    * ETH *leads* BTC at 1 s and 1 min, so BTC features help ETH little.
    * ETH is less predictable than BTC in IC terms, but it moves ~1.56× more, so its edge in bp
      is similar or larger.
    * The "large trade" threshold is 25 ETH (`LARGE_QTY`), like-for-like with 1 BTC.
    * Details are in `docs/FINDINGS.md` §0.

## 4. Commands (`make help`)
```
make install            pip install -e .
make test               unit tests (fast, no data needed)
make live VARIANT=spot  live forecaster (VARIANT=full needs Binance futures access)
make replay T="2026-10-15 13:37"   one forecast from local historical data
make data / features / train / calibrate / reports   full rebuild (~20 GB, a few hours)
make parity             live feature & score path == training (run after ANY feature change)
make skew               live-at-forecast-time vs final data on the real market
make refresh            monthly retrain on new data
```
`data/sec` can be a symlink to the per-second store shared with btc-predict: the directory names
are the same (`eth`, `ethum`, `spot`, `um`, alts, `eth_premium`, `ethusdc`, `coinbase/`).

## 5. Golden rules
* **No look-ahead.** Features use seconds ≤ T−1 s only. `y_*` / `px_*` columns are targets, never
  features.
* **One code path.** Change features only in `ethpredict/features/`; live reuses them. Then run
  `make test` and `make parity`.
* **Booster column order = `meta.json` `features` list.** Never reorder feature lists for
  existing models.
* **Never use the holdout to choose anything.** Decide on DEV months, confirm on confirmation
  months, then report the holdout once.
* **Mechanism first, and do not copy BTC conclusions.** Every kept signal has a market reason,
  is stable month by month, and was re-tested *on ETH*.
* **Be honest about costs.** Edges are ~0.5–1 bp versus 4–10 bp taker fees. This is a
  forecasting signal, not a standalone trading bot.
* **`LARGE_QTY` must match the stored data.** Changing it means rebuilding `data/sec/ethum`.
* **Shell hygiene.** Don't `pkill -f <pattern>` when that pattern is in your own command line;
  kill by PID.

## 6. How to do common tasks
* **Add a feature:**
  1. Implement it in `features/` (array-based, using seconds ≤ `end` only).
  2. Rebuild the matrices (`scripts/pipeline/build_features.py`).
  3. Test it with `scripts/research/run_model.py` on DEV months against a baseline, and compare
     with `scripts/research/compare_preds.py`.
  4. If it wins, holds on the confirmation months and is stable month by month, add it to the
     family definitions in `scripts/research/make_family_sets.py` and regenerate
     `configs/feature_sets`.
  5. Run `make train calibrate parity`, and update the README tables.
* **Test a mechanism ETH vs BTC:** copy a script from `scripts/research/mechanisms/`. They read
  both assets from the shared per-second store, pre-holdout days only.
* **Add a horizon:** add its target (see `features/subminute.py` for seconds), then run
  `train_final.py --horizons <h>` and `add_regime_calibration.py`. The predictor and CLI pick it
  up from `meta.json`.
* **Add a live data source:** implement it in `RestSource`, `ReplaySource` *and* `CachedSource`
  in `live/sources.py`, then run `make parity` and `make skew`.

## 7. Key results to know (holdout, full model; details in README)
RESULTS_PLACEHOLDER

## 8. Known limits
* The full variant could not be live-tested from the build environment (futures REST
  geo-blocked).
* Bands under-cover in strongly trending sessions.
* The ETH perp target is only weakly predictable, because the ETH perp is where price discovery
  happens.
