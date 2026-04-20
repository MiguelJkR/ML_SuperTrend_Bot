"""
ML SuperTrend v51 Trading Bot - Configuration v2
=================================================
Multi-broker configuration: OANDA (Forex/Gold) + Binance/Bybit (Crypto).
v2: Optimized parameters for profitability.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List

# ============================================================
# OANDA API CREDENTIALS - LIVE
# ============================================================
OANDA_LIVE = {
    "token": "YOUR_OANDA_LIVE_TOKEN",
    "account_id": "YOUR_OANDA_LIVE_ACCOUNT_ID",
    "url": "https://api-fxtrade.oanda.com",
    "stream_url": "https://stream-fxtrade.oanda.com",
}

# OANDA API CREDENTIALS - DEMO (Practice)
OANDA_DEMO = {
    "token": "YOUR_OANDA_DEMO_TOKEN",
    "account_id": "AUTO",
    "url": "https://api-fxpractice.oanda.com",
    "stream_url": "https://stream-fxpractice.oanda.com",
}

# ============================================================
# BINANCE FUTURES CREDENTIALS
# ============================================================
BINANCE_CONFIG = {
    "api_key": os.getenv("BINANCE_API_KEY", ""),
    "api_secret": os.getenv("BINANCE_API_SECRET", ""),
    "testnet": True,
}

# ============================================================
# BYBIT V5 CREDENTIALS
# ============================================================
BYBIT_CONFIG = {
    "api_key": os.getenv("BYBIT_API_KEY", ""),
    "api_secret": os.getenv("BYBIT_API_SECRET", ""),
    "testnet": True,
}

# ============================================================
# BROKER REGISTRY
# ============================================================
BROKER_CONFIGS = {
    "OANDA": OANDA_DEMO,
    # "BINANCE": BINANCE_CONFIG,
    # "BYBIT": BYBIT_CONFIG,
}

# Trading mode
LIVE_TRADING_ENABLED = True
DEMO_TRADING_ENABLED = True
DEMO_CAPITAL_LIMIT = 200.0

# TELEGRAM NOTIFICATIONS
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")
TELEGRAM_ENABLED = True

# ============================================================
# INSTRUMENT CONFIGURATIONS
# ============================================================
@dataclass
class InstrumentConfig:
    symbol: str
    timeframe: str
    is_xau: bool
    auto_long_only: bool
    auto_sl: float
    auto_be_tr: float
    auto_trail_tr: float
    auto_trail_m: float
    risk_pct: float = 0.04  # Aggressive: 4% base risk per trade
    max_units: int = 100000
    min_units: int = 1
    max_spread_pips: float = 3.0
    be_offset: float = 0.00020
    use_fixed_tp: bool = True
    tp_rr_ratio: float = 2.0
    tp_partial_close: float = 0.5  # Close 50% at 1R profit (aggressive mode)
    broker: str = "OANDA"

# --- FOREX (OANDA) ---
# H1: Wider SL (3.5 ATR), lower BE trigger (3.5), higher R:R (2.5:1)
# These fixes address the 82% SL-hit rate in backtests
EUR_USD_H1 = InstrumentConfig(
    symbol="EUR_USD", timeframe="H1", is_xau=False, auto_long_only=False,
    auto_sl=3.5, auto_be_tr=3.5, auto_trail_tr=1.5, auto_trail_m=0.8,
    risk_pct=0.015, be_offset=0.00200,  # Reduced risk per trade from 2% to 1.5%
    use_fixed_tp=True, tp_rr_ratio=2.5, broker="OANDA",
)

# M30: Already profitable (+26.62%), minor tuning only
EUR_USD_M30 = InstrumentConfig(
    symbol="EUR_USD", timeframe="M30", is_xau=False, auto_long_only=False,
    auto_sl=2.5, auto_be_tr=3.0, auto_trail_tr=1.0, auto_trail_m=0.7,
    risk_pct=0.04, be_offset=0.00150,  # Aggressive: 4% risk
    use_fixed_tp=True, tp_rr_ratio=2.0, broker="OANDA",
)

EUR_USD_M15 = InstrumentConfig(
    symbol="EUR_USD", timeframe="M15", is_xau=False, auto_long_only=False,
    auto_sl=2.0, auto_be_tr=3.0, auto_trail_tr=0.8, auto_trail_m=0.6,
    risk_pct=0.03, be_offset=0.00100,  # Aggressive: 3% risk (M15 smaller)
    use_fixed_tp=True, tp_rr_ratio=2.0, broker="OANDA",
)

# --- GOLD (OANDA) ---
XAU_USD_H1 = InstrumentConfig(
    symbol="XAU_USD", timeframe="H1", is_xau=True, auto_long_only=False,
    auto_sl=2.0, auto_be_tr=2.5, auto_trail_tr=2.5, auto_trail_m=1.5,
    risk_pct=0.015, be_offset=1.00, max_spread_pips=5.0, min_units=1,
    use_fixed_tp=True, tp_rr_ratio=2.0, broker="OANDA",
)

XAU_USD_M30 = InstrumentConfig(
    symbol="XAU_USD", timeframe="M30", is_xau=True, auto_long_only=False,
    auto_sl=2.0, auto_be_tr=2.0, auto_trail_tr=1.5, auto_trail_m=1.2,
    risk_pct=0.01, be_offset=0.50, max_spread_pips=5.0,
    use_fixed_tp=True, tp_rr_ratio=2.0, broker="OANDA",
)

# --- CRYPTO (BINANCE / BYBIT) ---
BTC_USDT_H1 = InstrumentConfig(
    symbol="BTC_USDT", timeframe="H1", is_xau=False, auto_long_only=False,
    auto_sl=2.5, auto_be_tr=3.0, auto_trail_tr=1.5, auto_trail_m=1.0,
    risk_pct=0.02, be_offset=50.0,
    max_spread_pips=10.0, max_units=10, min_units=1,
    use_fixed_tp=True, tp_rr_ratio=2.0, broker="BINANCE",
)

BTC_USDT_M30 = InstrumentConfig(
    symbol="BTC_USDT", timeframe="M30", is_xau=False, auto_long_only=False,
    auto_sl=2.0, auto_be_tr=2.5, auto_trail_tr=1.0, auto_trail_m=0.8,
    risk_pct=0.015, be_offset=30.0,
    max_spread_pips=10.0, max_units=10, min_units=1,
    use_fixed_tp=True, tp_rr_ratio=2.0, broker="BINANCE",
)

ETH_USDT_H1 = InstrumentConfig(
    symbol="ETH_USDT", timeframe="H1", is_xau=False, auto_long_only=False,
    auto_sl=2.5, auto_be_tr=3.0, auto_trail_tr=1.5, auto_trail_m=1.0,
    risk_pct=0.02, be_offset=5.0,
    max_spread_pips=5.0, max_units=100, min_units=1,
    use_fixed_tp=True, tp_rr_ratio=2.0, broker="BINANCE",
)

SOL_USDT_H1 = InstrumentConfig(
    symbol="SOL_USDT", timeframe="H1", is_xau=False, auto_long_only=False,
    auto_sl=3.0, auto_be_tr=3.5, auto_trail_tr=1.5, auto_trail_m=1.0,
    risk_pct=0.015, be_offset=0.50,
    max_spread_pips=3.0, max_units=1000, min_units=1,
    use_fixed_tp=True, tp_rr_ratio=2.5, broker="BINANCE",
)

# --- ADDITIONAL FOREX PAIRS (OANDA) ---
GBP_USD_M30 = InstrumentConfig(
    symbol="GBP_USD", timeframe="M30", is_xau=False, auto_long_only=False,
    auto_sl=2.5, auto_be_tr=3.0, auto_trail_tr=1.0, auto_trail_m=0.7,
    risk_pct=0.04, be_offset=0.00150,
    use_fixed_tp=True, tp_rr_ratio=2.0, broker="OANDA",
    max_spread_pips=3.5,
)

GBP_USD_M15 = InstrumentConfig(
    symbol="GBP_USD", timeframe="M15", is_xau=False, auto_long_only=False,
    auto_sl=2.0, auto_be_tr=3.0, auto_trail_tr=0.8, auto_trail_m=0.6,
    risk_pct=0.03, be_offset=0.00100,
    use_fixed_tp=True, tp_rr_ratio=2.0, broker="OANDA",
    max_spread_pips=3.5,
)

USD_JPY_M30 = InstrumentConfig(
    symbol="USD_JPY", timeframe="M30", is_xau=False, auto_long_only=False,
    auto_sl=2.5, auto_be_tr=3.0, auto_trail_tr=1.0, auto_trail_m=0.7,
    risk_pct=0.04, be_offset=0.150,  # JPY pairs have larger pip values
    use_fixed_tp=True, tp_rr_ratio=2.0, broker="OANDA",
    max_spread_pips=3.0,
)

USD_JPY_M15 = InstrumentConfig(
    symbol="USD_JPY", timeframe="M15", is_xau=False, auto_long_only=False,
    auto_sl=2.0, auto_be_tr=3.0, auto_trail_tr=0.8, auto_trail_m=0.6,
    risk_pct=0.03, be_offset=0.100,
    use_fixed_tp=True, tp_rr_ratio=2.0, broker="OANDA",
    max_spread_pips=3.0,
)

# ============================================================
# ACTIVE INSTRUMENTS -- Multi-pair aggressive day trading
# ============================================================
ACTIVE_INSTRUMENTS: List[InstrumentConfig] = [
    EUR_USD_M30,       # PRIMARY -- profitable (54.2% WR, PF 1.59)
    EUR_USD_M15,       # More signals on faster timeframe
    GBP_USD_M30,       # High volatility pair -- great for day trading
    GBP_USD_M15,       # Fast GBP signals
    USD_JPY_M30,       # Uncorrelated to EUR -- diversification
    USD_JPY_M15,       # Fast JPY signals
    # EUR_USD_H1,      # DISABLED: poor backtest results
    # XAU_USD_H1,      # Disabled: min 1 unit = too large for $200
    # XAU_USD_M30,     # Disabled: reactivate when capital >= $1,000
]

# ============================================================
# STRATEGY PARAMETERS v2 -- Optimized for Profitability
# ============================================================
@dataclass
class StrategyParams:
    # --- Core SuperTrend ---
    atr_len: int = 10
    atr_period: int = 10
    supertrend_factor: float = 3.0
    training_period: int = 200
    n_clusters: int = 3
    high_vol_pct: float = 0.75
    mid_vol_pct: float = 0.50
    low_vol_pct: float = 0.25
    
    # --- EMA filter ---
    use_ema: bool = True
    ema_fast: int = 15
    ema_slow: int = 50
    ema_fast_period: int = 15
    ema_slow_period: int = 50
    
    # --- ADX filter ---
    use_adx: bool = True
    adx_len: int = 14
    adx_period: int = 14
    adx_min: int = 15        # Lowered from 20 to 15 (was blocking borderline ADX 17-21)
    
    # --- MACD filter ---
    use_macd: bool = True
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # --- Session filter ---
    use_session: bool = True
    session_start: int = 6   # Widened from 7 to 6 (catch London open)
    session_end: int = 20    # Widened from 17 to 20 (catch US session)
    
    # --- Regime filter ---
    use_regime: bool = True
    regime_adx_period: int = 10
    regime_adx_min: float = 18.0
    
    # --- Major trend ---
    use_major_trend: bool = False
    major_ema_len: int = 200
    sma_period: int = 50
    rsi_period: int = 14
    
    # --- MA 21/50 crossover ---
    use_ma_cross: bool = True
    ma_cross_lookback: int = 10
    
    # --- Anti-revenge ---
    use_anti_revenge: bool = True
    max_consec_losses: int = 2
    pause_bars_after_loss: int = 4   # Reduced from 8 to 4 (less restrictive cooldown)
    min_bars_between: int = 3        # Reduced from 5 to 3 (allow more signal opportunities)
    
    # ========== NEW v2 QUALITY FILTERS ==========
    
    # MINIMUM SIGNAL STRENGTH -- reject weak signals (CRITICAL FIX)
    min_signal_strength: float = 0.40  # Lowered from 0.55 to 0.40 (was too strict with penalties)
    
    # RSI FILTER -- prevent buying overbought, selling oversold
    use_rsi_filter: bool = True
    rsi_overbought: float = 72.0   # Don't BUY if RSI > 72
    rsi_oversold: float = 28.0     # Don't SELL if RSI < 28
    
    # VOLATILITY SPIKE FILTER -- skip during abnormal ATR spikes
    use_volatility_filter: bool = True
    atr_spike_threshold: float = 3.0  # Widened from 2.0 to 3.0 (was blocking valid signals)
    
    # SUPERTREND DISTANCE FILTER -- reject entries too far from SuperTrend
    use_distance_filter: bool = True
    max_st_distance_atr: float = 3.0  # Widened from 2.0 to 3.0 (less restrictive)
    
    # CONFIRMATION CANDLE -- candle must close in signal direction
    use_confirmation_candle: bool = True
    
    # ADAPTIVE TP -- stronger signals get higher R:R
    use_adaptive_tp: bool = True  # Base RR + (strength-0.5)*1.0
    
    # ========== NEW v3 ADVANCED FILTERS ==========
    
    # BOLLINGER BANDS FILTER -- detect squeeze/breakout, overbought bands
    use_bb_filter: bool = True
    bb_period: int = 20
    bb_std_mult: float = 2.0
    bb_squeeze_bonus: float = 0.12    # Strength bonus when BB squeeze detected
    bb_outside_penalty: float = 0.15  # Penalty if price outside bands against signal
    
    # VOLUME FILTER -- confirm signals with volume above average
    use_volume_filter: bool = True
    vol_sma_period: int = 20
    vol_min_ratio: float = 0.7       # Minimum volume ratio (vs SMA) to NOT penalize
    vol_high_bonus: float = 0.10     # Bonus if volume > 1.5x average
    vol_low_penalty: float = 0.08    # Penalty if volume < 0.7x average
    
    # MULTI-TIMEFRAME SUPERTREND -- require H1 alignment for M15/M30 trades
    use_mtf_hard_filter: bool = True  # Hard block if H1 disagrees
    mtf_h1_weight: float = 0.15      # Strength bonus when H1 agrees
    
    # ENHANCED RSI -- divergence detection
    use_rsi_divergence: bool = True
    rsi_div_bonus: float = 0.12      # Bonus for favorable divergence
    rsi_div_penalty: float = 0.15    # Penalty for adverse divergence
    
    # ENHANCED MACD -- crossover confirmation
    use_macd_cross: bool = True
    macd_cross_lookback: int = 5     # Look for recent crossover in last N bars
    macd_cross_bonus: float = 0.12   # Bonus for recent favorable crossover

STRATEGY = StrategyParams()

# ============================================================
# RISK & POSITION LIMITS -- Aggressive mode
# ============================================================
MAX_CONCURRENT_POSITIONS = 3       # Up to 3 trades open simultaneously
MAX_POSITIONS_PER_PAIR = 1         # Max 1 trade per symbol
MAX_DAILY_TRADES = 10              # Safety cap: max 10 trades per day
MAX_DAILY_LOSS_PCT = 12.0          # Stop trading if daily loss > 12%

POLL_INTERVAL = 30
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 5000
LOG_FILE = "ml_supertrend_bot.log"
LOG_LEVEL = "INFO"

# ============================================================
# USER TIMEZONE (for Telegram alerts & day-of-week display)
# ============================================================
USER_TZ_OFFSET = -4   # UTC-4 (Eastern US / Colombia / Bolivia)
DAILY_SUMMARY_HOUR = 16  # 4 PM local = 20:00 UTC
