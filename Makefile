# One command per workflow. Run from the repo root.  `make help` lists them.
PY      ?= python3
START   ?= 2025-05-01
END     ?= $(shell date -u -d yesterday +%F)
VARIANT ?= spot
T       ?= 2026-10-15 13:37

.PHONY: help install test live replay data features train calibrate reports parity skew refresh clean-cache

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'

install:         ## install dependencies (package in editable mode)
	$(PY) -m pip install -e .

test:            ## unit tests (no market data needed)
	$(PY) -m pytest -q tests

live:            ## run the live forecaster (VARIANT=spot|full)
	$(PY) -m ethpredict.live --variant $(VARIANT)

replay:          ## one forecast for a historical minute from local data (T="YYYY-MM-DD HH:MM")
	$(PY) -m ethpredict.live --variant $(VARIANT) --replay "$(T)"

data:            ## download all raw data (~20 GB, ~45 min)
	$(PY) -m ethpredict.data.binance --start $(START) --end $(END) --workers 8 \
	  --jobs eth,ethum,eth_premium,ethusdc,spot,um,sol,xrp,doge,bnb,ada,avax,link,ltc,trx,sui,pepe,shib,dot,ton,near,bch
	$(PY) -m ethpredict.data.coinbase --start $(START)

features:        ## per-second bars -> minute panel, targets, model matrices
	$(PY) -m ethpredict.features.panel --start $(START) --end $(END)
	$(PY) -m ethpredict.features.subminute $(START) $(END)
	$(PY) scripts/pipeline/build_features.py $(START) $(END)

train:           ## holdout walk-forward + production models (~3 h)
	$(PY) scripts/pipeline/train_final.py --variants full,spot

calibrate:       ## regime-aware bands/hit tables + compress models for git
	$(PY) scripts/pipeline/add_regime_calibration.py
	for f in models/*.txt; do gzip -9 -k -f "$$f"; done

reports:         ## results table, charts, edge-vs-costs from cached holdout predictions
	$(PY) scripts/reports/results_table.py holdout
	$(PY) scripts/reports/make_report.py --pattern "final_full_{key}_holdout" --keys s1,f1,s5,f5,s2,f2 --tag holdout
	$(PY) scripts/reports/make_report.py --pattern "final_full_{key}_holdout" --keys s10s,f10s,s15s,f15s --tag holdout_sub
	$(PY) scripts/reports/edge_vs_costs.py s10s,f10s,s15s,f15s,s1,f1,s5,f5

parity:          ## prove live feature path == training features and scores
	$(PY) scripts/live/check_live_parity.py 5

skew:            ## live-at-forecast-time vs final data, on the real market (~10 min)
	$(PY) scripts/live/live_skew_now.py 5 $(VARIANT) 2.0

refresh:         ## monthly: new data -> features -> retrain production models
	scripts/pipeline/monthly_refresh.sh

clean-cache:     ## remove regenerable caches (keeps data/sec raw bars)
	rm -rf data/features reports/preds logs/*.json
