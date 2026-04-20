"""
Technical Indicators Module for ML SuperTrend v51 Trading Bot

This module provides core technical indicators with ML-enhanced volatility clustering.
All functions replicate PineScript logic in Python using numpy and pandas.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Average True Range (ATR) using RMA smoothing.
    
    Matches PineScript's ta.atr() using Running Moving Average smoothing.
    ATR measures volatility using the highest true range over the period.
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        period: Lookback period (default 10)
    
    Returns:
        numpy array of ATR values
    """
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)
    
    # Calculate true range
    tr1 = highs - lows
    tr2 = np.abs(highs - np.roll(closes, 1))
    tr3 = np.abs(lows - np.roll(closes, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # Handle first value
    
    # Apply RMA smoothing (alpha = 1/period)
    atr_result = rma(tr, period)
    return atr_result


def ema(data: np.ndarray, period: int) -> np.ndarray:
    """
    Exponential Moving Average (EMA).
    
    Calculates EMA with alpha = 2 / (period + 1).
    
    Args:
        data: Input data array
        period: Lookback period
    
    Returns:
        numpy array of EMA values
    """
    data = np.asarray(data, dtype=np.float64)
    alpha = 2.0 / (period + 1)
    ema_result = np.zeros_like(data)
    ema_result[0] = data[0]
    
    for i in range(1, len(data)):
        ema_result[i] = alpha * data[i] + (1 - alpha) * ema_result[i - 1]
    
    return ema_result


def sma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Simple Moving Average (SMA).
    
    Calculates the simple arithmetic mean over the period.
    
    Args:
        data: Input data array
        period: Lookback period
    
    Returns:
        numpy array of SMA values
    """
    data = np.asarray(data, dtype=np.float64)
    sma_result = np.zeros_like(data)
    
    for i in range(period - 1, len(data)):
        sma_result[i] = np.mean(data[i - period + 1:i + 1])
    
    return sma_result


def rma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Running Moving Average (RMA) - PineScript's ta.rma.
    
    RMA uses alpha = 1/period for smoothing.
    Also known as Wilder's moving average.
    
    Args:
        data: Input data array
        period: Lookback period
    
    Returns:
        numpy array of RMA values
    """
    data = np.asarray(data, dtype=np.float64)
    alpha = 1.0 / period
    rma_result = np.zeros_like(data)
    rma_result[0] = data[0]
    
    for i in range(1, len(data)):
        rma_result[i] = alpha * data[i] + (1 - alpha) * rma_result[i - 1]
    
    return rma_result


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index (RSI).
    
    Measures momentum using RMA smoothing of gains and losses.
    
    Args:
        closes: Array of close prices
        period: Lookback period (default 14)
    
    Returns:
        numpy array of RSI values
    """
    closes = np.asarray(closes, dtype=np.float64)
    deltas = np.diff(closes, prepend=closes[0])
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = rma(gains, period)
    avg_loss = rma(losses, period)
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_result = 100 - (100 / (1 + rs))
    
    return rsi_result


def adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Average Directional Index (ADX).
    
    Measures trend strength using directional movement (DI+ and DI-).
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        period: Lookback period (default 14)
    
    Returns:
        numpy array of ADX values
    """
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)
    
    # Calculate directional movements
    up_move = highs - np.roll(highs, 1)
    down_move = np.roll(lows, 1) - lows
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate true range
    tr1 = highs - lows
    tr2 = np.abs(highs - np.roll(closes, 1))
    tr3 = np.abs(lows - np.roll(closes, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Calculate directional indicators
    atr_val = rma(tr, period)
    plus_di = 100 * rma(plus_dm, period) / (atr_val + 1e-10)
    minus_di = 100 * rma(minus_dm, period) / (atr_val + 1e-10)
    
    # Calculate ADX
    di_diff = np.abs(plus_di - minus_di)
    di_sum = plus_di + minus_di
    di_ratio = di_diff / (di_sum + 1e-10)
    
    adx_result = rma(di_ratio * 100, period)
    
    return adx_result


def macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD (Moving Average Convergence Divergence).
    
    Returns MACD line, signal line, and histogram.
    
    Args:
        closes: Array of close prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    closes = np.asarray(closes, dtype=np.float64)
    
    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def kmeans_volatility_clustering(
    atr_values: np.ndarray,
    closes: np.ndarray,
    training_period: int = 200,
    n_clusters: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-Means clustering for volatility-based factor adaptation.
    
    Clusters normalized ATR values to identify low, mid, and high volatility regimes.
    Returns cluster labels and adaptive volatility factor for each candle.
    
    Args:
        atr_values: Array of ATR values
        closes: Array of close prices
        training_period: Number of candles for training (default 200)
        n_clusters: Number of clusters (default 3)
    
    Returns:
        Tuple of (cluster_labels, adaptive_factors)
    """
    atr_values = np.asarray(atr_values, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)
    
    # Normalize ATR by close price
    normalized_atr = atr_values / (closes + 1e-10)
    
    # Use training period data for K-Means initialization
    training_data = normalized_atr[:min(training_period, len(normalized_atr))]
    
    if len(training_data) < n_clusters:
        # Fallback if insufficient data
        return np.zeros(len(atr_values)), np.ones(len(atr_values))
    
    # Initialize centroids randomly from data
    np.random.seed(42)
    init_indices = np.random.choice(len(training_data), n_clusters, replace=False)
    centroids = training_data[init_indices].copy()
    
    # K-Means iterations (10 iterations for convergence)
    for _ in range(10):
        # Assign clusters
        distances = np.abs(training_data[:, np.newaxis] - centroids)
        assignments = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([
            np.mean(training_data[assignments == k]) if np.any(assignments == k) else centroids[k]
            for k in range(n_clusters)
        ])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    # Sort clusters by centroid (ascending: low volatility to high)
    sorted_indices = np.argsort(centroids)
    centroid_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
    
    # Assign all data points to clusters
    all_distances = np.abs(normalized_atr[:, np.newaxis] - centroids)
    all_assignments = np.argmin(all_distances, axis=1)
    cluster_labels = np.array([centroid_mapping[label] for label in all_assignments])
    
    # Define volatility factors (base_factor * (1 - percentage))
    base_factor = 2.0
    low_vol_pct = 0.75
    mid_vol_pct = 0.50
    high_vol_pct = 0.25
    
    factor_map = {
        0: base_factor * (1 - low_vol_pct),      # Low volatility cluster
        1: base_factor * (1 - mid_vol_pct),      # Mid volatility cluster
        2: base_factor * (1 - high_vol_pct)      # High volatility cluster
    }
    
    adaptive_factors = np.array([factor_map.get(label, base_factor) for label in cluster_labels])
    
    return cluster_labels, adaptive_factors


def supertrend(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr_values: np.ndarray,
    factor: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SuperTrend indicator with dynamic bands.
    
    Uses ATR-based upper and lower bands to determine trend direction.
    Direction: 1 = bullish, -1 = bearish.
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        atr_values: Array of ATR values
        factor: Multiplier for ATR bands
    
    Returns:
        Tuple of (supertrend_line, direction)
    """
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)
    atr_values = np.asarray(atr_values, dtype=np.float64)
    
    hl2 = (highs + lows) / 2.0
    basic_ub = hl2 + factor * atr_values
    basic_lb = hl2 - factor * atr_values
    
    # Calculate final bands (accounting for trend)
    final_ub = np.zeros_like(basic_ub)
    final_lb = np.zeros_like(basic_lb)
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    for i in range(1, len(basic_ub)):
        final_ub[i] = basic_ub[i] if basic_ub[i] < final_ub[i - 1] or closes[i - 1] > final_ub[i - 1] else final_ub[i - 1]
        final_lb[i] = basic_lb[i] if basic_lb[i] > final_lb[i - 1] or closes[i - 1] < final_lb[i - 1] else final_lb[i - 1]
    
    # Determine direction and SuperTrend line
    supertrend_line = np.zeros_like(closes)
    direction = np.zeros_like(closes)
    direction[0] = 1  # Start bullish
    supertrend_line[0] = final_lb[0]
    
    for i in range(1, len(closes)):
        if direction[i - 1] == 1:  # Was bullish
            if closes[i] <= final_ub[i]:
                direction[i] = 1
                supertrend_line[i] = final_lb[i]
            else:
                direction[i] = -1
                supertrend_line[i] = final_ub[i]
        else:  # Was bearish
            if closes[i] >= final_lb[i]:
                direction[i] = -1
                supertrend_line[i] = final_ub[i]
            else:
                direction[i] = 1
                supertrend_line[i] = final_lb[i]
    
    return supertrend_line, direction


def compute_all_indicators(candles: List[Dict[str, Any]], params: Any) -> Dict[str, Any]:
    """
    Master function computing all technical indicators.
    
    Takes a list of candle dictionaries and strategy parameters,
    returns a comprehensive dictionary of all computed indicators.
    
    Args:
        candles: List of candle dicts with 'open', 'high', 'low', 'close', 'time'
        params: StrategyParams object with indicator configuration
    
    Returns:
        Dictionary with all computed indicators
    """
    if not candles or len(candles) == 0:
        return {}
    
    # Extract OHLC data
    opens = np.array([c['open'] for c in candles], dtype=np.float64)
    highs = np.array([c['high'] for c in candles], dtype=np.float64)
    lows = np.array([c['low'] for c in candles], dtype=np.float64)
    closes = np.array([c['close'] for c in candles], dtype=np.float64)
    
    # Compute all indicators
    atr_values = atr(highs, lows, closes, getattr(params, 'atr_period', 10))
    ema_fast = ema(closes, getattr(params, 'ema_fast_period', 12))
    ema_slow = ema(closes, getattr(params, 'ema_slow_period', 26))
    sma_val = sma(closes, getattr(params, 'sma_period', 50))
    rsi_val = rsi(closes, getattr(params, 'rsi_period', 14))
    adx_val = adx(highs, lows, closes, getattr(params, 'adx_period', 14))
    macd_line, signal_line, histogram = macd(closes)
    
    # ML clustering and adaptive SuperTrend
    cluster_labels, adaptive_factors = kmeans_volatility_clustering(
        atr_values, closes,
        getattr(params, 'training_period', 200),
        getattr(params, 'n_clusters', 3)
    )
    
    base_factor = getattr(params, 'supertrend_factor', 3.0)
    supertrend_line, direction = supertrend(
        highs, lows, closes, atr_values, base_factor
    )
    
    # MA 21/50 crossover signals
    ma21 = ema(closes, 21)
    ma50 = ema(closes, 50)
    # Crossover detection: 1 = golden cross (21 crosses above 50), -1 = death cross
    ma_cross = np.zeros_like(closes)
    for i in range(1, len(closes)):
        if ma21[i] > ma50[i] and ma21[i-1] <= ma50[i-1]:
            ma_cross[i] = 1   # Golden cross (bullish)
        elif ma21[i] < ma50[i] and ma21[i-1] >= ma50[i-1]:
            ma_cross[i] = -1  # Death cross (bearish)

    # ==================== NEW INDICATORS v3 ====================
    
    # --- Bollinger Bands (20-period SMA, 2 std dev) ---
    bb_period = getattr(params, 'bb_period', 20)
    bb_std_mult = getattr(params, 'bb_std_mult', 2.0)
    bb_middle = sma(closes, bb_period)
    bb_std = np.zeros_like(closes)
    for i in range(bb_period - 1, len(closes)):
        bb_std[i] = np.std(closes[i - bb_period + 1:i + 1])
    bb_upper = bb_middle + bb_std_mult * bb_std
    bb_lower = bb_middle - bb_std_mult * bb_std
    # Bollinger Band Width (normalized): measures volatility squeeze
    bb_width = np.where(bb_middle > 0, (bb_upper - bb_lower) / bb_middle, 0)
    # %B: position of price within bands (0 = lower, 1 = upper, >1 = above upper)
    bb_pct_b = np.where((bb_upper - bb_lower) > 0,
                        (closes - bb_lower) / (bb_upper - bb_lower), 0.5)
    # BB squeeze detection: width < 20-period SMA of width * 0.75
    bb_width_sma = sma(bb_width, bb_period)
    bb_squeeze = np.where(bb_width < bb_width_sma * 0.75, 1.0, 0.0)
    
    # --- Volume Analysis ---
    volumes = np.array([c.get('volume', 0) for c in candles], dtype=np.float64)
    vol_sma_period = getattr(params, 'vol_sma_period', 20)
    vol_sma = sma(volumes, vol_sma_period)
    # Volume ratio: current volume / average volume (>1 = above average)
    vol_ratio = np.where(vol_sma > 0, volumes / vol_sma, 1.0)
    
    # --- MACD Crossover Detection ---
    macd_cross = np.zeros_like(closes)
    for i in range(1, len(closes)):
        if macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]:
            macd_cross[i] = 1   # Bullish crossover
        elif macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]:
            macd_cross[i] = -1  # Bearish crossover
    
    # --- Advanced Divergence Detection (RSI + MACD) ---
    # Uses swing pivot detection for accurate divergence identification.
    # Returns: regular divergence (+1 bullish / -1 bearish) and
    #          hidden divergence (+1 bullish continuation / -1 bearish continuation)
    rsi_divergence = np.zeros_like(closes)
    macd_divergence = np.zeros_like(closes)
    hidden_divergence = np.zeros_like(closes)
    divergence_strength = np.zeros_like(closes)  # 0.0 to 1.0 confidence

    # --- Find swing pivots (local min/max with N-bar confirmation) ---
    pivot_order = 5  # Bars on each side to confirm pivot
    swing_lows = []   # (index, price, rsi, macd_hist)
    swing_highs = []  # (index, price, rsi, macd_hist)

    for i in range(pivot_order, len(closes) - pivot_order):
        # Swing low: price is lowest in window of +/-pivot_order bars
        window_lo = lows[i - pivot_order:i + pivot_order + 1]
        if lows[i] == np.min(window_lo):
            swing_lows.append((i, lows[i], rsi_val[i], histogram[i]))
        # Swing high: price is highest in window
        window_hi = highs[i - pivot_order:i + pivot_order + 1]
        if highs[i] == np.max(window_hi):
            swing_highs.append((i, highs[i], rsi_val[i], histogram[i]))

    # --- Detect divergences between consecutive swing points ---
    div_max_bars = 50   # Max bars between pivots for valid divergence
    rsi_min_diff = 2.0  # Minimum RSI difference to count

    # RSI + MACD REGULAR BULLISH: price lower low, indicator higher low
    for j in range(1, len(swing_lows)):
        idx_prev, price_prev, rsi_prev, macd_prev = swing_lows[j - 1]
        idx_curr, price_curr, rsi_curr, macd_curr = swing_lows[j]
        bar_dist = idx_curr - idx_prev
        if bar_dist > div_max_bars or bar_dist < 4:
            continue
        # Regular bullish: price lower low + RSI higher low
        if price_curr < price_prev and rsi_curr > rsi_prev + rsi_min_diff:
            conf = min(1.0, (rsi_curr - rsi_prev) / 15.0)  # Confidence by RSI gap
            rsi_divergence[idx_curr] = 1.0
            divergence_strength[idx_curr] = max(divergence_strength[idx_curr], conf)
        # MACD regular bullish
        if price_curr < price_prev and macd_curr > macd_prev:
            macd_divergence[idx_curr] = 1.0
        # Hidden bullish (continuation): price higher low + RSI lower low
        if price_curr > price_prev and rsi_curr < rsi_prev - rsi_min_diff:
            hidden_divergence[idx_curr] = 1.0

    # RSI + MACD REGULAR BEARISH: price higher high, indicator lower high
    for j in range(1, len(swing_highs)):
        idx_prev, price_prev, rsi_prev, macd_prev = swing_highs[j - 1]
        idx_curr, price_curr, rsi_curr, macd_curr = swing_highs[j]
        bar_dist = idx_curr - idx_prev
        if bar_dist > div_max_bars or bar_dist < 4:
            continue
        # Regular bearish: price higher high + RSI lower high
        if price_curr > price_prev and rsi_curr < rsi_prev - rsi_min_diff:
            conf = min(1.0, (rsi_prev - rsi_curr) / 15.0)
            rsi_divergence[idx_curr] = -1.0
            divergence_strength[idx_curr] = max(divergence_strength[idx_curr], conf)
        # MACD regular bearish
        if price_curr > price_prev and macd_curr < macd_prev:
            macd_divergence[idx_curr] = -1.0
        # Hidden bearish (continuation): price lower high + RSI higher high
        if price_curr < price_prev and rsi_curr > rsi_prev + rsi_min_diff:
            hidden_divergence[idx_curr] = -1.0

    # Extend divergence signal for N bars after detection (signal lingers)
    div_linger = 3
    for arr in [rsi_divergence, macd_divergence, hidden_divergence, divergence_strength]:
        temp = arr.copy()
        for i in range(len(arr)):
            if temp[i] != 0:
                for k in range(1, div_linger + 1):
                    if i + k < len(arr) and arr[i + k] == 0:
                        arr[i + k] = temp[i] * (1.0 - k * 0.25)  # Decay

    # --- Volume Profile (VPOC, Value Area High/Low) ---
    vp_lookback = getattr(params, 'vp_lookback', 50)
    vpoc = closes[-1]
    vah = closes[-1]
    val_price = closes[-1]
    vp_strength = 0.0

    try:
        vp_closes = closes[-vp_lookback:]
        vp_volumes = volumes[-vp_lookback:]
        vp_highs = highs[-vp_lookback:]
        vp_lows = lows[-vp_lookback:]

        if len(vp_closes) >= 20 and np.sum(vp_volumes) > 0:
            price_min = float(np.min(vp_lows))
            price_max = float(np.max(vp_highs))
            n_bins = 30
            if price_max > price_min:
                bin_edges = np.linspace(price_min, price_max, n_bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                vol_profile = np.zeros(n_bins)

                for j in range(len(vp_closes)):
                    bar_lo = vp_lows[j]
                    bar_hi = vp_highs[j]
                    bar_vol = max(vp_volumes[j], 1)
                    for b in range(n_bins):
                        if bin_edges[b + 1] >= bar_lo and bin_edges[b] <= bar_hi:
                            vol_profile[b] += bar_vol

                vpoc_idx = int(np.argmax(vol_profile))
                vpoc = float(bin_centers[vpoc_idx])

                total_vol = np.sum(vol_profile)
                target_vol = total_vol * 0.70
                accumulated = vol_profile[vpoc_idx]
                lo_idx, hi_idx = vpoc_idx, vpoc_idx

                while accumulated < target_vol and (lo_idx > 0 or hi_idx < n_bins - 1):
                    expand_lo = vol_profile[lo_idx - 1] if lo_idx > 0 else 0
                    expand_hi = vol_profile[hi_idx + 1] if hi_idx < n_bins - 1 else 0
                    if expand_lo >= expand_hi and lo_idx > 0:
                        lo_idx -= 1
                        accumulated += expand_lo
                    elif hi_idx < n_bins - 1:
                        hi_idx += 1
                        accumulated += expand_hi
                    else:
                        lo_idx -= 1
                        accumulated += expand_lo

                val_price = float(bin_edges[lo_idx])
                vah = float(bin_edges[hi_idx + 1])

                price_range = price_max - price_min
                if price_range > 0:
                    distance_to_vpoc = abs(closes[-1] - vpoc) / price_range
                    vp_strength = max(0.0, 1.0 - distance_to_vpoc * 3)
    except Exception:
        pass

    vol_trend = 0.0
    if len(volumes) >= 20:
        recent_vol = np.mean(volumes[-5:])
        older_vol = np.mean(volumes[-20:-10])
        if older_vol > 0:
            vol_trend = (recent_vol - older_vol) / older_vol

    return {
        'atr': atr_values,
        'ema_fast': ema_fast,
        'ema_slow': ema_slow,
        'sma': sma_val,
        'rsi': rsi_val,
        'adx': adx_val,
        'macd_line': macd_line,
        'macd_signal': signal_line,
        'macd_histogram': histogram,
        'cluster_labels': cluster_labels,
        'adaptive_factors': adaptive_factors,
        'supertrend_line': supertrend_line,
        'supertrend_direction': direction,
        'ma21': ma21,
        'ma50': ma50,
        'ma_cross': ma_cross,
        # --- NEW v3 indicators ---
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'bb_middle': bb_middle,
        'bb_width': bb_width,
        'bb_pct_b': bb_pct_b,
        'bb_squeeze': bb_squeeze,
        'volume': volumes,
        'vol_sma': vol_sma,
        'vol_ratio': vol_ratio,
        'macd_cross': macd_cross,
        'rsi_divergence': rsi_divergence,
        'macd_divergence': macd_divergence,
        'hidden_divergence': hidden_divergence,
        'divergence_strength': divergence_strength,
        # --- v5 Volume Profile ---
        'vpoc': vpoc,
        'vah': vah,
        'val': val_price,
        'vp_strength': vp_strength,
        'vol_trend': vol_trend,
    }
