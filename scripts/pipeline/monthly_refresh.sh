#!/bin/bash
# Monthly refresh: pull new days, rebuild features, retrain production models on all data.
# Calibration and the online-layer state are kept from the last full holdout evaluation;
# re-run `python scripts/pipeline/train_final.py` (2-3 h) every few months to re-validate them.
#   usage: scripts/pipeline/monthly_refresh.sh        (from the repo root)
set -e
cd "$(dirname "$0")/../.."
END=$(date -u -d "yesterday" +%F)
python -m ethpredict.data.binance --start 2025-05-01 --end "$END" --workers 8 \
  --jobs eth,ethum,eth_premium,ethusdc,spot,um,sol,xrp,doge,bnb,ada,avax,link,ltc,trx,sui,pepe,shib,dot,ton,near,bch
python -m ethpredict.data.coinbase --start 2025-05-01 --end "$(date -u +%F)"
python -m ethpredict.features.panel --start 2025-05-01 --end "$END"
python -m ethpredict.features.subminute 2025-05-01 "$END"
python scripts/pipeline/build_features.py 2025-05-01 "$END"
python scripts/pipeline/train_final.py --variants full,spot --horizons 10s,15s,1,2,3,4,5 --production_only
python scripts/pipeline/add_regime_calibration.py
for f in models/*.txt; do gzip -9 -k -f "$f"; done
python scripts/live/check_live_parity.py 5
