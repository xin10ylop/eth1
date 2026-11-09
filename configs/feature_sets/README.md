# Feature sets

Each file is one column name per line, selecting from the model matrix (`X` + `extra_features` +
`Xalt` + `Xcb` + `Xalt2`). `scripts/pipeline/train_final.py` reads them. The exact list each
production booster was trained on is also stored in `models/meta.json` (`targets.<key>.features`),
and that stored copy is what the live predictor uses.

All files here are **generated** by `scripts/research/make_family_sets.py` from feature
*families*, i.e. from mechanisms, not from hand-picked columns. Edit the family rules there, then
regenerate.

| file | used for | contents |
|---|---|---|
| `core.txt`, `core_fast.txt` | full model, core part | ETH spot + perp microstructure, basis, perp premium index, Lévy areas, minute state, calendar, BTC spot |
| `sat.txt`, `sat_fast.txt` | full model, satellite part | core + Coinbase ETH premium + BTC perp (`xp_*`, `x_basis_dev*`) + 5-coin swarm (`alt_*`) |
| `spotcore.txt`, `spotcore_fast.txt` | spot-only model, core part | core without anything derived from Binance futures |
| `spotsat.txt`, `spotsat_fast.txt` | spot-only model, satellite part | satellite without futures-derived columns (no BTC perp) |
| `research/*.txt` | DEV experiments | `base`, `core`, `sat_all` (every family), `loo_<family>` (leave one family out), lean variants, 5-minute decompositions |

Rules:
* `*_fast` sets drop slow "regime" features (24 h returns, hour-of-day, round numbers…). Which
  horizons use them is set by `train_final.py --fast_from` (see `docs/FINDINGS.md` §4).
* Families **left out for ETH** after testing (`docs/FINDINGS.md` §3): the ETH/USDC book
  (`book_*`) and the 17-coin swarm (`swarm2_*`). Neither added anything on top of the satellite.
  The live forecaster skips fetching them when no model needs them.
* BTC/FDUSD and ETH/FDUSD books are not used (FDUSD's market share collapsed in Feb 2026).
