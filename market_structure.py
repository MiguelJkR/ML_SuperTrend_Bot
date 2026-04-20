"""
Market Structure Analysis Module

This module provides tools for analyzing market regime and structural levels
in trading bot applications. It includes:
- Range vs Trend detection using Bollinger Bands and ADX
- Support/Resistance level identification using pivots and swing analysis
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RegimeDetector:
    """
    Detects market regime (RANGING, TRENDING, or BREAKOUT) using Bollinger Band
    Width (BBW) and ADX levels.
    
    Market regimes:
    - RANGING: Low volatility, ADX < 20, stable price action within bands
    - TRENDING: High ADX (> 25), strong directional movement
    - BREAKOUT: Transition state from squeeze (low BBW) to expansion (rising ADX)
    """
    
    def __init__(self, bb_period: int = 20, bb_std: float = 2.0):
        """
        Initialize RegimeDetector.
        
        Args:
            bb_period: Bollinger Band period (default 20)
            bb_std: Bollinger Band standard deviations (default 2.0)
        """
        self.bb_period = bb_period
        self.bb_std = bb_std
        logger.info(f"RegimeDetector initialized with period={bb_period}, std={bb_std}")
    
    def _calculate_bollinger_bands(
        self, closes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands: middle (SMA), upper, and lower bands.
        
        Args:
            closes: Array of closing prices
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = np.convolve(closes, np.ones(self.bb_period) / self.bb_period, 
                            mode='valid')
        
        # Pad to match length
        middle = np.concatenate([np.full(len(closes) - len(middle), np.nan), middle])
        
        # Calculate standard deviation for each window
        std = np.array([
            np.std(closes[max(0, i - self.bb_period + 1):i + 1])
            for i in range(len(closes))
        ])
        
        upper = middle + self.bb_std * std
        lower = middle - self.bb_std * std
        
        return upper, middle, lower
    
    def _calculate_bbw(self, closes: np.ndarray) -> np.ndarray:
        """
        Calculate Bollinger Band Width as percentage of middle band.
        
        BBW = (Upper - Lower) / Middle * 100
        
        Args:
            closes: Array of closing prices
            
        Returns:
            Array of BBW values (percentage)
        """
        upper, middle, lower = self._calculate_bollinger_bands(closes)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            bbw = np.where(middle != 0, (upper - lower) / middle * 100, 0)
        
        bbw = np.nan_to_num(bbw, nan=0, posinf=0, neginf=0)
        return bbw
    
    def _bbw_percentile(self, bbw_values: np.ndarray, lookback: int = 100) -> float:
        """
        Calculate BBW percentile rank within lookback period.
        
        Helps identify squeeze (low percentile) vs expansion (high percentile).
        
        Args:
            bbw_values: Array of BBW values
            lookback: Number of bars to consider for percentile
            
        Returns:
            Percentile rank (0-100) of current BBW
        """
        if len(bbw_values) < lookback:
            lookback = len(bbw_values)
        
        recent_bbw = bbw_values[-lookback:]
        current_bbw = bbw_values[-1]
        
        # Avoid division by zero
        if np.std(recent_bbw) == 0:
            return 50.0
        
        percentile = (np.sum(recent_bbw <= current_bbw) / len(recent_bbw)) * 100
        return float(percentile)
    
    def _adx_to_strength(self, adx: float) -> float:
        """
        Convert ADX value to trend strength (0-1 scale).
        
        ADX scale:
        - 0-20: Weak/No trend
        - 20-40: Moderate trend
        - 40-60: Strong trend
        - 60+: Very strong trend
        
        Args:
            adx: ADX value
            
        Returns:
            Normalized strength (0-1)
        """
        if adx <= 0:
            return 0.0
        elif adx >= 60:
            return 1.0
        else:
            return min(adx / 60, 1.0)
    
    def detect(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        adx_values: np.ndarray,
        lookback_bbw: int = 100
    ) -> Dict:
        """
        Detect market regime based on BBW and ADX levels.
        
        Args:
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of closing prices
            adx_values: Array of ADX values
            lookback_bbw: Lookback period for BBW percentile calculation
            
        Returns:
            Dictionary with keys:
            - 'regime': str, one of 'RANGING', 'TRENDING', 'BREAKOUT'
            - 'bbw': float, current Bollinger Band Width (percentage)
            - 'bbw_percentile': float, BBW percentile (0-100)
            - 'adx': float, current ADX value
            - 'strength': float, trend strength (0-1)
        """
        # Calculate BBW
        bbw_values = self._calculate_bbw(closes)
        current_bbw = bbw_values[-1]
        bbw_perc = self._bbw_percentile(bbw_values, lookback_bbw)
        
        # Current ADX
        current_adx = float(adx_values[-1]) if len(adx_values) > 0 else 20.0
        
        # Calculate strength
        strength = self._adx_to_strength(current_adx)
        
        # Determine regime
        regime = self._classify_regime(current_adx, bbw_perc)
        
        logger.info(
            f"Regime Detection - ADX: {current_adx:.2f}, BBW: {current_bbw:.2f}, "
            f"BBW%: {bbw_perc:.1f}, Regime: {regime}"
        )
        
        return {
            'regime': regime,
            'bbw': float(current_bbw),
            'bbw_percentile': float(bbw_perc),
            'adx': current_adx,
            'strength': strength
        }
    
    def _classify_regime(self, adx: float, bbw_percentile: float) -> str:
        """
        Classify regime based on ADX and BBW percentile.
        
        Args:
            adx: ADX value
            bbw_percentile: BBW percentile rank
            
        Returns:
            Regime string: 'RANGING', 'TRENDING', or 'BREAKOUT'
        """
        # RANGING: Low ADX + low BBW percentile (squeeze)
        if adx < 20 and bbw_percentile < 40:
            return 'RANGING'
        
        # TRENDING: High ADX
        elif adx >= 25:
            return 'TRENDING'
        
        # BREAKOUT: Transition from squeeze to expansion
        # (historically low BBW but now expanding with rising ADX)
        elif bbw_percentile > 60 and 20 <= adx < 25:
            return 'BREAKOUT'
        
        # Default to ranging if uncertain
        else:
            return 'RANGING'


class HTFSRLevels:
    """
    Higher Timeframe (Weekly/Daily) Support & Resistance analyzer.
    Fetches W1 and D1 candles from OANDA to identify key structural levels
    that complement the intraday S/R analysis.
    
    These levels are stronger than intraday pivots because they represent
    institutional price memory across larger time horizons.
    """
    
    def __init__(self, client=None):
        """Initialize with optional OANDA client for data fetching."""
        self.client = client
        self.weekly_levels: Dict = {}
        self.daily_levels: Dict = {}
        self._last_update: Optional[str] = None
        logger.info("HTFSRLevels initialized")
    
    def update(self, symbol: str) -> Dict:
        """
        Fetch weekly and daily candles, compute key S/R levels.
        
        Returns dict with:
        - weekly_high/low: Previous week's range
        - daily_high/low: Previous day's range  
        - weekly_pivots: Standard pivots from weekly bar
        - daily_pivots: Standard pivots from daily bar
        - key_levels: Merged and sorted list of all significant levels
        """
        if not self.client:
            return {}
        
        result = {'weekly': {}, 'daily': {}, 'key_levels': []}
        
        try:
            # Fetch last 10 weekly candles
            weekly = self.client.get_candles(symbol, 'W', count=10)
            if weekly and len(weekly) >= 2:
                prev_w = weekly[-2]  # Completed weekly bar
                w_pivot = (prev_w['high'] + prev_w['low'] + prev_w['close']) / 3
                w_r1 = 2 * w_pivot - prev_w['low']
                w_s1 = 2 * w_pivot - prev_w['high']
                
                result['weekly'] = {
                    'high': prev_w['high'],
                    'low': prev_w['low'],
                    'pivot': w_pivot,
                    'r1': w_r1,
                    's1': w_s1,
                }
                result['key_levels'].extend([prev_w['high'], prev_w['low'], w_pivot, w_r1, w_s1])
                
                # Also add 4-week high/low
                if len(weekly) >= 5:
                    four_w_high = max(c['high'] for c in weekly[-5:-1])
                    four_w_low = min(c['low'] for c in weekly[-5:-1])
                    result['weekly']['four_week_high'] = four_w_high
                    result['weekly']['four_week_low'] = four_w_low
                    result['key_levels'].extend([four_w_high, four_w_low])
            
            # Fetch last 10 daily candles
            daily = self.client.get_candles(symbol, 'D', count=10)
            if daily and len(daily) >= 2:
                prev_d = daily[-2]  # Completed daily bar
                d_pivot = (prev_d['high'] + prev_d['low'] + prev_d['close']) / 3
                d_r1 = 2 * d_pivot - prev_d['low']
                d_s1 = 2 * d_pivot - prev_d['high']
                
                result['daily'] = {
                    'high': prev_d['high'],
                    'low': prev_d['low'],
                    'pivot': d_pivot,
                    'r1': d_r1,
                    's1': d_s1,
                }
                result['key_levels'].extend([prev_d['high'], prev_d['low'], d_pivot, d_r1, d_s1])
            
            # Sort and deduplicate key levels
            result['key_levels'] = sorted(set(round(l, 5) for l in result['key_levels']))
            self._last_update = symbol
            
            logger.info(f"HTF S/R for {symbol}: {len(result['key_levels'])} key levels, "
                       f"Weekly pivot={result['weekly'].get('pivot', 0):.5f}, "
                       f"Daily pivot={result['daily'].get('pivot', 0):.5f}")
            
        except Exception as e:
            logger.warning(f"HTF S/R update failed for {symbol}: {e}")
        
        return result
    
    def get_nearest_htf_levels(self, key_levels: List[float], current_price: float, 
                                atr: float) -> Dict:
        """
        Find nearest HTF support and resistance from key levels.
        
        Returns:
            Dict with htf_resistance, htf_support, and ATR-normalized distances
        """
        if not key_levels or atr <= 0:
            return {'htf_resistance': 0, 'htf_support': 0, 
                    'htf_distance_to_r': float('inf'), 'htf_distance_to_s': float('inf')}
        
        above = [l for l in key_levels if l > current_price]
        below = [l for l in key_levels if l < current_price]
        
        htf_r = min(above) if above else current_price * 1.01
        htf_s = max(below) if below else current_price * 0.99
        
        return {
            'htf_resistance': htf_r,
            'htf_support': htf_s,
            'htf_distance_to_r': (htf_r - current_price) / atr,
            'htf_distance_to_s': (current_price - htf_s) / atr,
        }


class SRLevels:
    """
    Identifies Support and Resistance levels using multiple methods:
    - Pivot points (Standard and Fibonacci)
    - Swing highs/lows detection
    - Level clustering
    - ATR-normalized distance calculations
    """
    
    def __init__(self):
        """Initialize SRLevels analyzer."""
        logger.info("SRLevels initialized")
    
    def _calculate_pivot_points(
        self, high: float, low: float, close: float, method: str = 'standard'
    ) -> Dict[str, float]:
        """
        Calculate pivot points using Standard or Fibonacci method.
        
        Standard Pivot Points:
        - Pivot = (H + L + C) / 3
        - R1 = (2 * Pivot) - Low
        - S1 = (2 * Pivot) - High
        - R2/S2, R3/S3 are extensions
        
        Fibonacci Pivot Points:
        - Uses Fibonacci ratios (0.382, 0.618, 1.0)
        
        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close
            method: 'standard' or 'fibonacci'
            
        Returns:
            Dictionary with keys: pivot, r1, r2, r3, s1, s2, s3
        """
        pivot = (high + low + close) / 3
        
        if method == 'fibonacci':
            # Fibonacci levels
            hl_range = high - low
            r1 = pivot + 0.382 * hl_range
            r2 = pivot + 0.618 * hl_range
            r3 = pivot + 1.0 * hl_range
            s1 = pivot - 0.382 * hl_range
            s2 = pivot - 0.618 * hl_range
            s3 = pivot - 1.0 * hl_range
        else:
            # Standard pivot points
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = r2 + (high - low)
            s3 = s2 - (high - low)
        
        return {
            'pivot': float(pivot),
            'r1': float(r1),
            'r2': float(r2),
            'r3': float(r3),
            's1': float(s1),
            's2': float(s2),
            's3': float(s3)
        }
    
    def _find_swing_points(
        self, highs: np.ndarray, lows: np.ndarray, lookback: int = 5
    ) -> Tuple[List[float], List[float]]:
        """
        Detect swing highs and swing lows using local min/max.
        
        A swing high is a local maximum over lookback bars.
        A swing low is a local minimum over lookback bars.
        
        Args:
            highs: Array of high prices
            lows: Array of low prices
            lookback: Number of bars for local extrema detection
            
        Returns:
            Tuple of (swing_highs, swing_lows) lists
        """
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(highs) - lookback):
            window_high = highs[i - lookback:i + lookback + 1]
            window_low = lows[i - lookback:i + lookback + 1]
            
            # Check if current is local maximum or minimum
            if highs[i] == np.max(window_high):
                swing_highs.append(float(highs[i]))
            
            if lows[i] == np.min(window_low):
                swing_lows.append(float(lows[i]))
        
        return swing_highs, swing_lows
    
    def _cluster_levels(self, levels: List[float], tolerance: float) -> List[float]:
        """
        Cluster nearby levels together and return cluster centroids.
        
        Levels within tolerance distance are grouped together.
        
        Args:
            levels: List of price levels
            tolerance: Minimum distance between distinct levels
            
        Returns:
            List of clustered level centroids
        """
        if not levels:
            return []
        
        if len(levels) == 1:
            return levels
        
        sorted_levels = sorted(levels)
        clusters = [[sorted_levels[0]]]
        
        for level in sorted_levels[1:]:
            # Check if level belongs to existing cluster
            if abs(level - np.mean(clusters[-1])) <= tolerance:
                clusters[-1].append(level)
            else:
                clusters.append([level])
        
        # Return cluster means
        return [np.mean(cluster) for cluster in clusters]
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, 
                      closes: np.ndarray, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR).
        
        Args:
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of closing prices
            period: ATR period (default 14)
            
        Returns:
            Current ATR value
        """
        # True Range = max(H-L, abs(H-Prev_C), abs(L-Prev_C))
        hl = highs[-period:] - lows[-period:]
        hc = np.abs(highs[-period:] - closes[-period - 1:-1])
        lc = np.abs(lows[-period:] - closes[-period - 1:-1])
        
        tr = np.maximum(hl, np.maximum(hc, lc))
        atr = np.mean(tr)
        
        return float(atr)
    
    def calculate(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        current_price: Optional[float] = None,
        lookback: int = 50,
        swing_lookback: int = 5,
        cluster_tolerance: Optional[float] = None
    ) -> Dict:
        """
        Calculate support and resistance levels.
        
        Args:
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of closing prices
            current_price: Current price (if None, uses last close)
            lookback: Lookback period for pivot calculation (uses last bar)
            swing_lookback: Lookback for swing point detection
            cluster_tolerance: Distance tolerance for level clustering
                              (if None, uses 0.1% of current price)
            
        Returns:
            Dictionary with keys:
            - 'pivot': float, pivot point
            - 'r1', 'r2', 'r3': float, resistance levels
            - 's1', 's2', 's3': float, support levels
            - 'swing_highs': list, detected swing high prices
            - 'swing_lows': list, detected swing low prices
            - 'nearest_resistance': float, closest resistance above price
            - 'nearest_support': float, closest support below price
            - 'distance_to_resistance_atr': float, distance to R in ATR units
            - 'distance_to_support_atr': float, distance to S in ATR units
        """
        if current_price is None:
            current_price = closes[-1]
        
        # Auto-calculate clustering tolerance if not provided
        if cluster_tolerance is None:
            cluster_tolerance = current_price * 0.001  # 0.1%
        
        # Calculate pivot points from previous bar
        if len(closes) > 1:
            prev_high = highs[-2]
            prev_low = lows[-2]
            prev_close = closes[-2]
        else:
            prev_high = highs[-1]
            prev_low = lows[-1]
            prev_close = closes[-1]
        
        pivots = self._calculate_pivot_points(prev_high, prev_low, prev_close)
        
        # Find swing points
        swing_highs, swing_lows = self._find_swing_points(
            highs, lows, swing_lookback
        )
        
        # Cluster swing points
        clustered_highs = self._cluster_levels(swing_highs, cluster_tolerance)
        clustered_lows = self._cluster_levels(swing_lows, cluster_tolerance)
        
        # Combine all resistance and support levels
        resistance_levels = [pivots['r1'], pivots['r2'], pivots['r3']]
        resistance_levels.extend([h for h in clustered_highs if h > current_price])
        resistance_levels = sorted(set(resistance_levels))
        
        support_levels = [pivots['s1'], pivots['s2'], pivots['s3']]
        support_levels.extend([l for l in clustered_lows if l < current_price])
        support_levels = sorted(set(support_levels), reverse=True)
        
        # Find nearest levels
        nearest_resistance = next(
            (r for r in resistance_levels if r > current_price),
            None
        ) or max(resistance_levels) if resistance_levels else current_price * 1.01
        
        nearest_support = next(
            (s for s in support_levels if s < current_price),
            None
        ) or min(support_levels) if support_levels else current_price * 0.99
        
        # Calculate ATR
        atr = self._calculate_atr(highs, lows, closes)
        
        # Calculate ATR-normalized distances
        distance_to_resistance_atr = (nearest_resistance - current_price) / atr \
            if atr > 0 else 0
        distance_to_support_atr = (current_price - nearest_support) / atr \
            if atr > 0 else 0
        
        logger.info(
            f"SR Levels - Price: {current_price:.2f}, "
            f"Support: {nearest_support:.2f}, Resistance: {nearest_resistance:.2f}, "
            f"R dist (ATR): {distance_to_resistance_atr:.2f}, "
            f"S dist (ATR): {distance_to_support_atr:.2f}"
        )
        
        return {
            'pivot': pivots['pivot'],
            'r1': pivots['r1'],
            'r2': pivots['r2'],
            'r3': pivots['r3'],
            's1': pivots['s1'],
            's2': pivots['s2'],
            's3': pivots['s3'],
            'swing_highs': clustered_highs,
            'swing_lows': clustered_lows,
            'nearest_resistance': float(nearest_resistance),
            'nearest_support': float(nearest_support),
            'distance_to_resistance_atr': float(distance_to_resistance_atr),
            'distance_to_support_atr': float(distance_to_support_atr)
        }
