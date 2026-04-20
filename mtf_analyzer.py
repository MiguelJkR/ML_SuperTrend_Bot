"""
Multi-Timeframe (MTF) Analysis Module for Trading Bot

This module provides multi-timeframe analysis by fetching candles for higher
timeframes (H4, D1) alongside the primary H1 timeframe, computing SuperTrend
direction on each timeframe, and returning MTF confirmation signals.

Key Features:
    - Fetches candles for H1, H4, and D1 timeframes
    - Computes SuperTrend direction on each timeframe using consistent parameters
    - Returns MTF confirmation signal: does higher TF agree with H1 signal?
    - MTF score: 1.0 (both H4+D1 agree), 0.5 (only H4 agrees), 0.0 (neither agrees)
    - Comprehensive error handling and logging
"""

import logging
from typing import Dict, Optional, List, Any
from oanda_client import OandaClient
from indicators import compute_all_indicators
from config import StrategyParams, STRATEGY

logger = logging.getLogger(__name__)


class MTFAnalyzer:
    """
    Multi-Timeframe (MTF) analyzer for trading signals.

    Analyzes price action across multiple timeframes (H1, H4, D1) to confirm
    trading signals. Higher timeframe agreement improves signal reliability.

    Attributes:
        client (OandaClient): OANDA API client for fetching candles
        params (StrategyParams): Strategy parameters for indicator computation
    """

    def __init__(self, client: OandaClient, params: StrategyParams):
        """
        Initialize MTFAnalyzer with OANDA client and strategy parameters.

        Args:
            client (OandaClient): OANDA v20 API client instance
            params (StrategyParams): Strategy parameters object with indicator settings

        Raises:
            ValueError: If client or params is None
        """
        if client is None:
            raise ValueError("OandaClient instance cannot be None")
        if params is None:
            raise ValueError("StrategyParams instance cannot be None")

        self.client = client
        self.params = params
        logger.info("MTFAnalyzer initialized successfully")

    def _fetch_candles(
        self,
        symbol: str,
        granularity: str,
        count: int = 300
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch candles for a specific timeframe.

        Args:
            symbol (str): Currency pair (e.g., 'EUR_USD')
            granularity (str): Timeframe (H1, H4, D, etc.)
            count (int): Number of candles to fetch (default 300)

        Returns:
            List of candle dictionaries or None if fetch fails

        Logs:
            - Error if fetch fails or returns empty list
        """
        try:
            candles = self.client.get_candles(symbol, granularity, count=count)

            if candles is None or len(candles) == 0:
                logger.warning(
                    f"Failed to fetch candles for {symbol} {granularity}: "
                    f"empty result from API"
                )
                return None

            logger.debug(f"Fetched {len(candles)} candles for {symbol} {granularity}")
            return candles

        except Exception as e:
            logger.error(
                f"Exception while fetching {symbol} {granularity} candles: {e}",
                exc_info=True
            )
            return None

    def _compute_supertrend_direction(
        self,
        candles: List[Dict[str, Any]],
        symbol: str,
        granularity: str
    ) -> Optional[int]:
        """
        Compute SuperTrend direction for a given set of candles.

        Args:
            candles (List[Dict]): Candle data from OANDA
            symbol (str): Currency pair (for logging only)
            granularity (str): Timeframe (for logging only)

        Returns:
            SuperTrend direction: 1 (bullish), -1 (bearish), or None if computation fails

        Logs:
            - Error if compute_all_indicators fails or returns empty dict
            - Warning if supertrend_direction key missing
        """
        try:
            if not candles or len(candles) < 20:
                logger.warning(
                    f"Insufficient candles for {symbol} {granularity}: "
                    f"need at least 20, got {len(candles) if candles else 0}"
                )
                return None

            indicators = compute_all_indicators(candles, self.params)

            if not indicators or "supertrend_direction" not in indicators:
                logger.warning(
                    f"Failed to compute SuperTrend for {symbol} {granularity}: "
                    f"missing supertrend_direction in indicators"
                )
                return None

            # Extract the latest (most recent) SuperTrend direction
            direction_array = indicators["supertrend_direction"]
            latest_direction = int(direction_array[-1])

            logger.debug(
                f"SuperTrend direction for {symbol} {granularity}: {latest_direction}"
            )
            return latest_direction

        except Exception as e:
            logger.error(
                f"Exception computing SuperTrend for {symbol} {granularity}: {e}",
                exc_info=True
            )
            return None

    def get_htf_bias(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze higher timeframe bias and return MTF confirmation signal.

        Fetches candles for H1, H4, and D1 timeframes, computes SuperTrend
        direction on each, and returns a comprehensive MTF confirmation signal.

        Args:
            symbol (str): Currency pair (e.g., 'EUR_USD')

        Returns:
            Dictionary with keys:
                - h4_direction (int): H4 SuperTrend direction (1 or -1)
                - d1_direction (int): D1 SuperTrend direction (1 or -1)
                - h4_confirms_long (bool): True if H4 is bullish
                - h4_confirms_short (bool): True if H4 is bearish
                - d1_confirms_long (bool): True if D1 is bullish
                - d1_confirms_short (bool): True if D1 is bearish
                - mtf_score (float): Confirmation score:
                    * 1.0 if both H4 and D1 agree
                    * 0.5 if only H4 agrees (D1 unavailable/unclear)
                    * 0.0 if neither agrees or both unavailable

        Logs:
            - Error if all timeframe fetches fail (returns default dict with zeros)
            - Warning if individual timeframe fetch fails
            - Info level summary of computed bias

        Raises:
            No exceptions; all errors are logged and handled gracefully
        """
        try:
            logger.info(f"Analyzing MTF bias for {symbol}")

            # Fetch candles for each timeframe
            h1_candles = self._fetch_candles(symbol, "H1", count=300)
            h4_candles = self._fetch_candles(symbol, "H4", count=300)
            d1_candles = self._fetch_candles(symbol, "D", count=300)

            # Compute SuperTrend directions
            h1_direction = self._compute_supertrend_direction(h1_candles, symbol, "H1")
            h4_direction = self._compute_supertrend_direction(h4_candles, symbol, "H4")
            d1_direction = self._compute_supertrend_direction(d1_candles, symbol, "D1")

            # Handle missing data gracefully
            if h4_direction is None:
                logger.warning(f"H4 direction unavailable for {symbol}, defaulting to 0")
                h4_direction = 0
            if d1_direction is None:
                logger.warning(f"D1 direction unavailable for {symbol}, defaulting to 0")
                d1_direction = 0

            # Determine confirmation booleans
            h4_confirms_long = h4_direction == 1
            h4_confirms_short = h4_direction == -1
            d1_confirms_long = d1_direction == 1
            d1_confirms_short = d1_direction == -1

            # Calculate MTF score
            mtf_score = 0.0
            if h4_direction != 0 and d1_direction != 0:
                # Both timeframes available
                if h4_direction == d1_direction:
                    mtf_score = 1.0  # Both H4 and D1 agree
                else:
                    mtf_score = 0.0  # They disagree
            elif h4_direction != 0:
                # Only H4 available
                mtf_score = 0.5

            result = {
                'h1_direction': h1_direction if h1_direction is not None else 0,
                'h4_direction': h4_direction,
                'd1_direction': d1_direction,
                'h4_confirms_long': h4_confirms_long,
                'h4_confirms_short': h4_confirms_short,
                'd1_confirms_long': d1_confirms_long,
                'd1_confirms_short': d1_confirms_short,
                'mtf_score': mtf_score,
            }

            logger.info(
                f"MTF Bias for {symbol}: H4={h4_direction:+d}, D1={d1_direction:+d}, "
                f"Score={mtf_score:.1f}"
            )

            return result

        except Exception as e:
            logger.error(
                f"Unexpected error in get_htf_bias for {symbol}: {e}",
                exc_info=True
            )
            # Return safe default on unexpected error
            return {
                'h1_direction': 0,
                'h4_direction': 0,
                'd1_direction': 0,
                'h4_confirms_long': False,
                'h4_confirms_short': False,
                'd1_confirms_long': False,
                'd1_confirms_short': False,
                'mtf_score': 0.0,
            }
