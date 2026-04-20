"""
ML SuperTrend v51 - Correlation Manager
========================================
Prevents excessive exposure to correlated instruments.
If EUR_USD and GBP_USD both trigger LONG, the effective risk doubles because
they're ~0.85 correlated. This module limits total directional exposure
to correlated pairs and adjusts position sizing accordingly.

Usage:
    from correlation_manager import CorrelationManager
    cm = CorrelationManager(client)
    allowed, adj_risk = cm.check_trade_allowed("EUR_USD", "LONG", base_risk=0.04)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# =====================================================================
# STATIC CORRELATION GROUPS -- based on well-known FX correlations
# =====================================================================
# Group 1: USD-negative pairs (move together when USD weakens)
# Group 2: USD-positive pairs (move together when USD strengthens)
# Group 3: Commodities
CORRELATION_GROUPS = {
    "USD_NEGATIVE": {
        "instruments": ["EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD"],
        "base_correlation": 0.80,
        "description": "USD-negative FX pairs -- highly correlated"
    },
    "USD_POSITIVE": {
        "instruments": ["USD_JPY", "USD_CHF", "USD_CAD"],
        "base_correlation": 0.70,
        "description": "USD-positive FX pairs -- moderately correlated"
    },
    "COMMODITIES": {
        "instruments": ["XAU_USD", "XAG_USD"],
        "base_correlation": 0.85,
        "description": "Precious metals -- highly correlated"
    },
    "CRYPTO_MAJOR": {
        "instruments": ["BTC_USDT", "ETH_USDT"],
        "base_correlation": 0.75,
        "description": "Major crypto -- correlated in risk-on/off moves"
    },
}

# Inverse correlation pairs: when one goes LONG, the other tends to go SHORT
INVERSE_CORRELATIONS = {
    ("EUR_USD", "USD_JPY"): -0.40,   # Partially inverse via USD
    ("EUR_USD", "USD_CHF"): -0.85,   # Strongly inverse
    ("GBP_USD", "USD_JPY"): -0.35,   # Partially inverse
}


class CorrelationManager:
    """
    Manages correlation-based risk limits for multi-pair trading.

    Key features:
    1. Static correlation groups (pre-defined FX/crypto relationships)
    2. Dynamic correlation calculation from recent price data
    3. Maximum directional exposure per correlation group
    4. Position size adjustment based on existing correlated exposure
    """

    def __init__(
        self,
        oanda_client=None,
        max_group_exposure: float = 2.0,        # Max combined risk % per group
        max_same_direction: int = 2,             # Max same-direction trades in group
        correlation_threshold: float = 0.60,     # Min correlation to consider "related"
        use_dynamic_correlation: bool = True,     # Calculate from price data
        dynamic_lookback: int = 100,             # Bars for dynamic calculation
    ):
        self.client = oanda_client
        self.max_group_exposure = max_group_exposure
        self.max_same_direction = max_same_direction
        self.correlation_threshold = correlation_threshold
        self.use_dynamic_correlation = use_dynamic_correlation
        self.dynamic_lookback = dynamic_lookback

        # Track active exposures: {"EUR_USD": {"direction": "LONG", "risk_pct": 0.04}}
        self.active_exposures: Dict[str, Dict] = {}

        # Cache for dynamic correlations
        self._corr_cache: Dict[Tuple[str, str], float] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 3600  # Refresh hourly

        # Build lookup: instrument -> group name
        self._instrument_groups: Dict[str, str] = {}
        for group_name, group_data in CORRELATION_GROUPS.items():
            for inst in group_data["instruments"]:
                self._instrument_groups[inst] = group_name

        logger.info(f"CorrelationManager initialized: max_group_exposure={max_group_exposure}%, "
                     f"max_same_dir={max_same_direction}, threshold={correlation_threshold}")

    def update_exposure(self, instrument: str, direction: str, risk_pct: float):
        """Register an active trade's exposure."""
        symbol = self._normalize_symbol(instrument)
        self.active_exposures[symbol] = {
            "direction": direction.upper(),
            "risk_pct": risk_pct,
            "time": datetime.now(timezone.utc).isoformat()
        }
        logger.info(f"Correlation: registered {direction} {symbol} @ {risk_pct:.1%} risk")

    def remove_exposure(self, instrument: str):
        """Remove a closed trade's exposure."""
        symbol = self._normalize_symbol(instrument)
        if symbol in self.active_exposures:
            del self.active_exposures[symbol]
            logger.info(f"Correlation: removed exposure for {symbol}")

    def check_trade_allowed(
        self,
        instrument: str,
        direction: str,
        base_risk: float = 0.04,
    ) -> Tuple[bool, float, List[str]]:
        """
        Check if a new trade is allowed given current correlated exposures.

        Returns:
            (allowed, adjusted_risk, reasons)
            - allowed: True if trade can be taken
            - adjusted_risk: Risk % after correlation adjustment (may be reduced)
            - reasons: List of explanation strings
        """
        symbol = self._normalize_symbol(instrument)
        direction = direction.upper()
        reasons = []

        # 1. Find which group this instrument belongs to
        group_name = self._instrument_groups.get(symbol)
        if not group_name:
            # Not in any group -- no correlation restriction
            return True, base_risk, ["no_corr_group"]

        group = CORRELATION_GROUPS[group_name]
        group_instruments = group["instruments"]

        # 2. Find active trades in the same correlation group
        same_group_trades = []
        for inst, exp in self.active_exposures.items():
            if inst in group_instruments and inst != symbol:
                same_group_trades.append((inst, exp))

        if not same_group_trades:
            # First trade in this group -- full risk allowed
            return True, base_risk, ["first_in_group"]

        # 3. Count same-direction trades in group
        same_dir_count = sum(1 for _, exp in same_group_trades if exp["direction"] == direction)

        if same_dir_count >= self.max_same_direction:
            reasons.append(f"blocked:max_same_dir({same_dir_count}/{self.max_same_direction})")
            logger.warning(f"Correlation BLOCKED: {direction} {symbol} -- "
                          f"{same_dir_count} same-direction trades already in {group_name}")
            return False, 0, reasons

        # 4. Calculate total group exposure
        total_group_risk = sum(exp["risk_pct"] for _, exp in same_group_trades
                              if exp["direction"] == direction)

        remaining_budget = self.max_group_exposure / 100.0 - total_group_risk

        if remaining_budget <= 0.005:  # Less than 0.5% left
            reasons.append(f"blocked:group_limit({total_group_risk:.1%}/{self.max_group_exposure}%)")
            logger.warning(f"Correlation BLOCKED: {direction} {symbol} -- "
                          f"group {group_name} at {total_group_risk:.1%} exposure")
            return False, 0, reasons

        # 5. Check inverse correlations (opposite direction trades reduce risk)
        inverse_offset = 0
        for (a, b), inv_corr in INVERSE_CORRELATIONS.items():
            if symbol == a or symbol == b:
                other = b if symbol == a else a
                if other in self.active_exposures:
                    other_exp = self.active_exposures[other]
                    # If inversely correlated and opposite directions -> natural hedge
                    if other_exp["direction"] != direction and inv_corr < -0.3:
                        inverse_offset += other_exp["risk_pct"] * abs(inv_corr) * 0.5
                        reasons.append(f"hedge_offset:{other}({abs(inv_corr):.0%})")

        # 6. Adjust risk based on group concentration
        adjusted_risk = min(base_risk, remaining_budget + inverse_offset)

        # Scale down based on number of correlated trades
        concentration_factor = 1.0 / (1.0 + same_dir_count * 0.3)  # 1.0, 0.77, 0.63...
        adjusted_risk *= concentration_factor

        # Don't allow risk below 0.5% (not worth the trade)
        if adjusted_risk < 0.005:
            reasons.append(f"blocked:risk_too_small({adjusted_risk:.2%})")
            return False, 0, reasons

        if adjusted_risk < base_risk:
            reduction = (1 - adjusted_risk / base_risk) * 100
            reasons.append(f"risk_reduced:{reduction:.0f}%_due_to_{group_name}")
            logger.info(f"Correlation: {symbol} risk reduced {base_risk:.1%} -> {adjusted_risk:.1%} "
                       f"({same_dir_count} correlated trades in {group_name})")
        else:
            reasons.append("full_risk_allowed")

        return True, adjusted_risk, reasons

    def get_dynamic_correlation(self, symbol_a: str, symbol_b: str) -> Optional[float]:
        """
        Calculate rolling correlation between two instruments from recent price data.
        Uses OANDA candle closes over the lookback period.
        """
        if not self.client:
            return None

        # Check cache
        cache_key = tuple(sorted([symbol_a, symbol_b]))
        now = datetime.now(timezone.utc)
        if (self._cache_time and
            (now - self._cache_time).total_seconds() < self._cache_ttl_seconds and
            cache_key in self._corr_cache):
            return self._corr_cache[cache_key]

        try:
            candles_a = self.client.get_candles(symbol_a, "H1", count=self.dynamic_lookback)
            candles_b = self.client.get_candles(symbol_b, "H1", count=self.dynamic_lookback)

            if not candles_a or not candles_b:
                return None

            # Align by length
            min_len = min(len(candles_a), len(candles_b))
            closes_a = np.array([c["close"] for c in candles_a[-min_len:]])
            closes_b = np.array([c["close"] for c in candles_b[-min_len:]])

            # Calculate returns
            returns_a = np.diff(closes_a) / closes_a[:-1]
            returns_b = np.diff(closes_b) / closes_b[:-1]

            if len(returns_a) < 20:
                return None

            correlation = np.corrcoef(returns_a, returns_b)[0, 1]

            # Cache result
            self._corr_cache[cache_key] = correlation
            self._cache_time = now

            return correlation

        except Exception as e:
            logger.warning(f"Dynamic correlation failed for {symbol_a}/{symbol_b}: {e}")
            return None

    def get_group_status(self) -> List[Dict]:
        """Get current exposure status per correlation group for dashboard."""
        status = []
        for group_name, group_data in CORRELATION_GROUPS.items():
            group_trades = []
            total_risk = 0
            for inst in group_data["instruments"]:
                if inst in self.active_exposures:
                    exp = self.active_exposures[inst]
                    group_trades.append({
                        "instrument": inst,
                        "direction": exp["direction"],
                        "risk_pct": exp["risk_pct"]
                    })
                    total_risk += exp["risk_pct"]

            if group_trades:
                status.append({
                    "group": group_name,
                    "description": group_data["description"],
                    "trades": group_trades,
                    "total_risk_pct": total_risk,
                    "max_risk_pct": self.max_group_exposure / 100.0,
                    "utilization": total_risk / (self.max_group_exposure / 100.0) if self.max_group_exposure > 0 else 0,
                })
        return status

    def _normalize_symbol(self, instrument: str) -> str:
        """Normalize instrument name: EUR_USD_M30 -> EUR_USD"""
        parts = instrument.split("_")
        if len(parts) >= 2:
            # Check if last part is a timeframe
            if parts[-1] in ("M1", "M5", "M15", "M30", "H1", "H4", "D", "W"):
                return "_".join(parts[:-1])
        return instrument
