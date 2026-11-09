"""ethpredict: 10-second to 5-minute Ethereum forecasting (Binance ETH spot & USD-M perpetual).

A port of btc-predict; everything asset-specific lives in config.py (asset profile).

Pipeline stages = subpackages:
    data/      download raw market data           (Binance bulk files, Coinbase candles)
    features/  per-second bars -> features/targets (shared verbatim by training and live)
    modeling/  LightGBM models, online layer, calibration, walk-forward evaluation
    live/      real-time data sources, predictor, CLI  (python -m ethpredict.live)
    research/  experiments that were tested and NOT shipped (TCN, OKX)
"""

__version__ = "1.0.0"
