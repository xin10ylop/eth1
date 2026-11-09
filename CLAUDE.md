# CLAUDE.md

Read **AGENTS.md** first. It is the complete A-to-Z guide to this repository: reading order, repo
map, key concepts, commands and golden rules.

Quick rules:
* No look-ahead.
* Change features only in `ethpredict/features/`, then run `make test` and `make parity`.
* Never tune on the holdout months (2026-05 → 2026-10).
* Do not carry over btc-predict conclusions without re-testing them on ETH.
* Kill processes by PID, never with `pkill -f` on a pattern that is in your own command line.
