"""Every path, constant and the ASSET PROFILE used by the project live here.

The code base is a port of btc-predict. Everything asset-specific is defined in the profile
below, so the same machinery forecasts ETH: the target is ETH, and BTC plays the role ETH played
for Bitcoin (the other major: its spot price in the e_* features, its perpetual in the xp_*
features, and it is part of the alt "swarm").
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- asset profile ------------------------------------------------------------------------------
ASSET = "ETH"
SPOT_SYMBOL = "ETHUSDT"            # target spot (venue "s")
PERP_SYMBOL = "ETHUSDT"            # target USD-M perpetual (venue "f")
LEAD_SPOT_SYMBOL = "BTCUSDT"       # the other major, spot  -> e_* features ("external")
LEAD_PERP_SYMBOL = "BTCUSDT"       # the other major, perp  -> xp_* features ("cross perp")
COINBASE_PRODUCT = "ETH-USD"       # Coinbase premium features
TARGET_KEY, LEAD_KEY = "eth", "btc"

# per-second data directories (data/sec/<dir>/<day>.parquet); names shared with btc-predict
DIR_SPOT, DIR_PERP = "eth", "ethum"
DIR_LEAD_SPOT, DIR_LEAD_PERP = "spot", "um"
DIR_PREMIUM = "eth_premium"        # target perp premium-index 1m klines
DIR_DEPTH = None                   # aux order-book depth / open-interest: research-only, not used
DIR_METRICS = None
COINBASE_FILE = "coinbase_ETH-USD_1m.parquet"
# "large trade" threshold in base units used when aggregating perp trades into seconds (lbv/lsv).
# 25 ETH (~$75-100k) ~ the notional of 1 BTC used by btc-predict, i.e. genuinely large prints.
# live MUST use the same value. Changing it requires re-downloading the ethum per-second data.
LARGE_QTY = 25.0

# cross-asset universe: key -> (data dir, Binance spot symbol)
ALTS = ("btc", "sol", "xrp", "doge", "bnb")          # 5-coin swarm (first = the other major)
ALTS2 = ("ada", "avax", "link", "ltc", "trx", "sui", "pepe", "shib", "dot", "ton", "near", "bch")
BOOKS = ("ethusdc",)                                  # other books of the same asset (FDUSD excluded: venue collapsed Feb 2026)
SPOT_SYMBOLS = {"eth": "ETHUSDT", "btc": "BTCUSDT", "sol": "SOLUSDT", "xrp": "XRPUSDT", "doge": "DOGEUSDT",
                "bnb": "BNBUSDT", "ethusdc": "ETHUSDC", **{a: a.upper() + "USDT" for a in ALTS2}}
DATA_DIRS = {"eth": DIR_SPOT, "btc": DIR_LEAD_SPOT, **{k: k for k in ("sol", "xrp", "doge", "bnb", "ethusdc") + ALTS2}}

# ---- local data (not in git; rebuilt by the pipeline) --------------------------------------
DATA = ROOT / "data"
RAW = DATA / "raw"
SEC = DATA / "sec"              # per-second bars (can be a symlink to a store shared with btc-predict)
MIN = DATA / "min"
FEAT = DATA / "features"        # panel.parquet (minute panel + targets), X*.parquet, Ysub.parquet

# ---- versioned artifacts ----------------------------------------------------------------------
MODELS = ROOT / "models"
FEATURE_SETS = ROOT / "configs" / "feature_sets"
REPORTS = ROOT / "reports"
HOLDOUT = REPORTS / "holdout"
CONFIRM = REPORTS / "confirm"
FIGURES = REPORTS / "figures"
DEV = REPORTS / "dev"
RESEARCH = REPORTS / "research"
LIVE_REPORTS = REPORTS / "live"
PREDS = REPORTS / "preds"
LOGS = ROOT / "logs"

# ---- endpoints ---------------------------------------------------------------------------------
BINANCE_VISION = "https://data.binance.vision/data"
BINANCE_LIVE = "https://data-api.binance.vision/api/v3"

HORIZONS = (1, 2, 3, 4, 5)   # minute horizons; second horizons (10s, 15s) live in features/subminute.py

for p in (RAW, SEC, MIN, FEAT, MODELS, HOLDOUT, CONFIRM, FIGURES, DEV, RESEARCH, LIVE_REPORTS, PREDS):
    p.mkdir(parents=True, exist_ok=True)
