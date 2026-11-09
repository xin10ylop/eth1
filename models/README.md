# Production models

* `<variant>_<target>_<part>.txt.gz` are LightGBM boosters, gzip-compressed for git. The predictor
  reads `.txt` if present, otherwise `.txt.gz`.
  * `variant`: `full` (needs Binance USD-M futures) or `spot` (spot data + Coinbase only).
  * `target`: venue (`s` = spot ETHUSDT, `f` = perp ETHUSDT) + horizon (`10s`, `15s`, `1`…`5`
    minutes).
  * `part`: `core` or `sat` (satellite).
* `meta.json` holds everything else the live predictor needs.

```
meta[variant]:
  uses_futures, half_life, regime{vrel_window_min, band_edges, regime_edges, regime_names}
  targets[key]:
    venue, h (minutes, float), label ("10s", "1m" ...), files{core, sat}
    features{core: [...], sat: [...]}      # exact column order of each booster
    rls_state                              # online ensemble state at the end of the holdout
    calib [a, b]                           # P(up) = sigmoid(a + b z)
    beta                                   # E[r] = beta * z * sigma * sqrt(h)
    resid_q, band_table[vrel bin][q]       # standardized residual quantiles (0.025 .. 0.975)
    et_hour_band_factor[24], vr_edges, vr_band_factor[3]
    absz_quantiles, absz_deciles, hit_by_conf_decile, hit_table[decile][regime]
    holdout{core, sat, ens}                # holdout metrics (same as reports/holdout/*.json)
```

Rebuilt by `make train` (holdout + production) or `make refresh` (production only), then
`make calibrate`.
