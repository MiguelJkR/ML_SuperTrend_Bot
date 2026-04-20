from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np


@dataclass
class RegimeState:
    """Represents the current market regime and associated metrics."""
    regime: str  # "TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE"
    confidence: float  # 0.0 to 1.0
    adx_value: float
    atr_percentile: float
    bb_width_percentile: float
    trend_strength: float  # -1 (strong down) to +1 (strong up)
    volatility_state: str  # "LOW", "NORMAL", "HIGH", "EXTREME"
    recommended_strategy: str  # "TREND_FOLLOW", "MEAN_REVERT", "STAY_OUT"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        """Convert regime state to dictionary."""
        return {
            'regime': self.regime,
            'confidence': self.confidence,
            'adx_value': self.adx_value,
            'atr_percentile': self.atr_percentile,
            'bb_width_percentile': self.bb_width_percentile,
            'trend_strength': self.trend_strength,
            'volatility_state': self.volatility_state,
            'recommended_strategy': self.recommended_strategy,
            'timestamp': self.timestamp.isoformat()
        }


class MarketRegimeDetector:
    """
    Detects market regime (TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE)
    using ADX, Bollinger Bands, ATR, EMA, and Hurst Exponent estimation.
    """
    
    def __init__(
        self,
        adx_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        lookback: int = 100,
        ema21_period: int = 21,
        ema50_period: int = 50
    ):
        self.adx_period = adx_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.lookback = lookback
        self.ema21_period = ema21_period
        self.ema50_period = ema50_period
        
        self.closes = np.array([])
        self.highs = np.array([])
        self.lows = np.array([])
        self.volumes = np.array([])
        self.times = []
        
        self.regime_history = []
        self.last_regime = None
        
    def update(self, candles: List[Dict]) -> RegimeState:
        """
        Analyze candles and return current market regime.
        """
        closes_list = []
        highs_list = []
        lows_list = []
        volumes_list = []
        times_list = []
        
        for candle in candles:
            if 'mid' in candle and isinstance(candle['mid'], dict):
                close = float(candle['mid']['c'])
                high = float(candle['mid']['h'])
                low = float(candle['mid']['l'])
            else:
                close = float(candle.get('close', 0))
                high = float(candle.get('high', 0))
                low = float(candle.get('low', 0))
            volume = float(candle.get('volume', 0))
            
            closes_list.append(close)
            highs_list.append(high)
            lows_list.append(low)
            volumes_list.append(volume)
            times_list.append(candle.get('time', datetime.utcnow().isoformat()))
        
        self.closes = np.array(closes_list, dtype=np.float64)
        self.highs = np.array(highs_list, dtype=np.float64)
        self.lows = np.array(lows_list, dtype=np.float64)
        self.volumes = np.array(volumes_list, dtype=np.float64)
        self.times = times_list
        
        if len(self.closes) < self.adx_period + 10:
            return RegimeState(
                regime="VOLATILE",
                confidence=0.0,
                adx_value=0.0,
                atr_percentile=0.5,
                bb_width_percentile=0.5,
                trend_strength=0.0,
                volatility_state="NORMAL",
                recommended_strategy="STAY_OUT"
            )
        
        adx_value = self._calculate_adx()
        plus_di, minus_di = self._calculate_di()
        atr = self._calculate_atr()
        atr_percentile = self._calculate_atr_percentile(atr)
        
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands()
        bb_width = bb_upper - bb_lower
        bb_width_percentile = self._calculate_bb_width_percentile(bb_width)
        
        ema21 = self._calculate_ema(self.ema21_period)
        ema50 = self._calculate_ema(self.ema50_period)
        
        current_price = self.closes[-1]
        
        trend_strength = self._calculate_trend_strength(plus_di, minus_di)
        volatility_state = self._classify_volatility(atr_percentile, bb_width_percentile)
        
        regime, confidence = self._classify_regime(
            adx_value, plus_di, minus_di, current_price, ema21, ema50,
            atr_percentile, bb_width_percentile, volatility_state
        )
        
        recommended_strategy = self._get_recommended_strategy(regime, volatility_state, confidence)
        
        regime_state = RegimeState(
            regime=regime,
            confidence=confidence,
            adx_value=adx_value,
            atr_percentile=atr_percentile,
            bb_width_percentile=bb_width_percentile,
            trend_strength=trend_strength,
            volatility_state=volatility_state,
            recommended_strategy=recommended_strategy
        )
        
        if self.last_regime is None or self.last_regime != regime:
            self.regime_history.append({
                'timestamp': regime_state.timestamp.isoformat(),
                'regime': regime,
                'confidence': confidence,
                'adx': adx_value,
                'trend_strength': trend_strength,
                'volatility_state': volatility_state
            })
            self.last_regime = regime
            
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-1000:]
        
        return regime_state
    
    def _calculate_adx(self) -> float:
        """Calculate ADX (Average Directional Index)."""
        if len(self.highs) < self.adx_period + 5:
            return 0.0
        
        highs_diff = np.diff(self.highs)
        lows_diff = -np.diff(self.lows)
        
        plus_dm = np.zeros(len(highs_diff))
        minus_dm = np.zeros(len(lows_diff))
        
        for i in range(len(highs_diff)):
            if highs_diff[i] > lows_diff[i] and highs_diff[i] > 0:
                plus_dm[i] = highs_diff[i]
            if lows_diff[i] > highs_diff[i] and lows_diff[i] > 0:
                minus_dm[i] = lows_diff[i]
        
        tr = self._calculate_true_range()[1:]
        
        smoothed_tr = self._wilders_ma(tr, self.adx_period)
        smoothed_tr = np.where(smoothed_tr == 0, 1e-10, smoothed_tr)
        
        plus_di = self._wilders_ma(plus_dm, self.adx_period) / smoothed_tr * 100
        minus_di = self._wilders_ma(minus_dm, self.adx_period) / smoothed_tr * 100
        
        di_diff = np.abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        
        di_sum = np.where(di_sum == 0, 1e-10, di_sum)
        dx = di_diff / di_sum * 100
        
        adx = self._wilders_ma(dx, self.adx_period)
        
        return float(adx[-1]) if len(adx) > 0 else 0.0
    
    def _calculate_di(self) -> tuple:
        """Calculate +DI and -DI."""
        if len(self.highs) < self.adx_period + 5:
            return 0.0, 0.0
        
        highs_diff = np.diff(self.highs)
        lows_diff = -np.diff(self.lows)
        
        plus_dm = np.zeros(len(highs_diff))
        minus_dm = np.zeros(len(lows_diff))
        
        for i in range(len(highs_diff)):
            if highs_diff[i] > lows_diff[i] and highs_diff[i] > 0:
                plus_dm[i] = highs_diff[i]
            if lows_diff[i] > highs_diff[i] and lows_diff[i] > 0:
                minus_dm[i] = lows_diff[i]
        
        tr = self._calculate_true_range()[1:]
        atr = np.mean(tr[-self.adx_period:]) if len(tr) >= self.adx_period else np.mean(tr) if len(tr) > 0 else 1e-10
        atr = max(atr, 1e-10)
        
        plus_di_raw = self._wilders_ma(plus_dm, self.adx_period) / atr * 100
        minus_di_raw = self._wilders_ma(minus_dm, self.adx_period) / atr * 100
        
        plus_di = float(plus_di_raw[-1]) if len(plus_di_raw) > 0 else 0.0
        minus_di = float(minus_di_raw[-1]) if len(minus_di_raw) > 0 else 0.0
        
        return plus_di, minus_di
    
    def _calculate_true_range(self) -> np.ndarray:
        """Calculate True Range."""
        tr1 = self.highs - self.lows
        tr2 = np.abs(self.highs - np.append(self.closes[0], self.closes[:-1]))
        tr3 = np.abs(self.lows - np.append(self.closes[0], self.closes[:-1]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return tr
    
    def _calculate_atr(self) -> np.ndarray:
        """Calculate ATR (Average True Range)."""
        tr = self._calculate_true_range()
        atr = self._wilders_ma(tr, self.atr_period)
        return atr
    
    def _wilders_ma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Wilder's moving average (exponential with alpha = 1/period)."""
        if len(data) < period:
            return data
        
        result = np.zeros(len(data))
        result[period - 1] = np.mean(data[:period])
        
        alpha = 1.0 / period
        for i in range(period, len(data)):
            result[i] = result[i - 1] * (1 - alpha) + data[i] * alpha
        
        return result
    
    def _calculate_bollinger_bands(self) -> tuple:
        """Calculate Bollinger Bands (upper, middle, lower)."""
        if len(self.closes) < self.bb_period:
            return self.closes[-1], self.closes[-1], self.closes[-1]
        
        sma = self._simple_ma(self.closes, self.bb_period)
        std = np.std(self.closes[-self.bb_period:])
        
        middle = sma[-1]
        upper = middle + self.bb_std * std
        lower = middle - self.bb_std * std
        
        return float(upper), float(middle), float(lower)
    
    def _simple_ma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate simple moving average."""
        if len(data) < period:
            return data
        
        result = np.convolve(data, np.ones(period) / period, mode='valid')
        padding = len(data) - len(result)
        result = np.concatenate([np.full(padding, np.nan), result])
        return result
    
    def _calculate_ema(self, period: int) -> float:
        """Calculate EMA at current price."""
        if len(self.closes) < period:
            return self.closes[-1]
        
        ema = self._exponential_ma(self.closes, period)
        return float(ema[-1])
    
    def _exponential_ma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate exponential moving average."""
        if len(data) < period:
            return data
        
        alpha = 2.0 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = ema[i - 1] * (1 - alpha) + data[i] * alpha
        
        return ema
    
    def _calculate_atr_percentile(self, current_atr: float) -> float:
        """Calculate ATR percentile relative to lookback period."""
        atr = self._calculate_atr()
        
        if len(atr) < self.lookback:
            lookback_atr = atr
        else:
            lookback_atr = atr[-self.lookback:]
        
        if len(lookback_atr) == 0:
            return 0.5
        
        current_atr_val = float(atr[-1])
        count_below = np.sum(lookback_atr <= current_atr_val)
        percentile = count_below / len(lookback_atr)
        
        return float(np.clip(percentile, 0.0, 1.0))
    
    def _calculate_bb_width_percentile(self, current_bb_width: float) -> float:
        """Calculate Bollinger Band width percentile."""
        if len(self.closes) < self.bb_period + self.lookback:
            return 0.5
        
        bb_widths = []
        
        for i in range(len(self.closes) - self.lookback, len(self.closes)):
            if i + 1 < self.bb_period:
                continue
            
            sma = np.mean(self.closes[i - self.bb_period + 1:i + 1])
            std = np.std(self.closes[i - self.bb_period + 1:i + 1])
            bb_w = 2 * self.bb_std * std
            bb_widths.append(bb_w)
        
        if len(bb_widths) == 0:
            return 0.5
        
        bb_widths = np.array(bb_widths)
        count_below = np.sum(bb_widths <= current_bb_width)
        percentile = count_below / len(bb_widths)
        
        return float(np.clip(percentile, 0.0, 1.0))
    
    def _calculate_trend_strength(self, plus_di: float, minus_di: float) -> float:
        """Calculate trend strength from -1 (strong down) to +1 (strong up)."""
        di_sum = plus_di + minus_di
        if di_sum < 1e-10:
            return 0.0
        
        trend_strength = (plus_di - minus_di) / di_sum
        return float(np.clip(trend_strength, -1.0, 1.0))
    
    def _classify_volatility(self, atr_percentile: float, bb_width_percentile: float) -> str:
        """Classify volatility state based on ATR and BB width percentiles."""
        if atr_percentile > 0.75 or bb_width_percentile > 0.75:
            return "EXTREME"
        elif atr_percentile > 0.55 or bb_width_percentile > 0.55:
            return "HIGH"
        elif atr_percentile < 0.25 or bb_width_percentile < 0.25:
            return "LOW"
        else:
            return "NORMAL"
    
    def _classify_regime(
        self,
        adx_value: float,
        plus_di: float,
        minus_di: float,
        current_price: float,
        ema21: float,
        ema50: float,
        atr_percentile: float,
        bb_width_percentile: float,
        volatility_state: str
    ) -> tuple:
        """Classify market regime and confidence."""
        if volatility_state in ["EXTREME", "HIGH"] and adx_value < 20:
            confidence = 0.6 + atr_percentile * 0.3
            return "VOLATILE", confidence
        
        if adx_value > 25:
            if plus_di > minus_di and current_price > ema50:
                confidence = 0.4 + (adx_value - 25) / 30 * 0.4
                confidence = min(0.95, confidence)
                return "TRENDING_UP", confidence
            
            if minus_di > plus_di and current_price < ema50:
                confidence = 0.4 + (adx_value - 25) / 30 * 0.4
                confidence = min(0.95, confidence)
                return "TRENDING_DOWN", confidence
        
        if adx_value < 20 and volatility_state in ["LOW", "NORMAL"]:
            confidence = 0.5 + (1 - atr_percentile) * 0.4
            return "RANGING", confidence
        
        if 20 <= adx_value <= 25:
            if plus_di > minus_di and current_price > ema21:
                confidence = 0.3 + (adx_value - 20) / 5 * 0.3
                return "TRENDING_UP", confidence
            elif minus_di > plus_di and current_price < ema21:
                confidence = 0.3 + (adx_value - 20) / 5 * 0.3
                return "TRENDING_DOWN", confidence
        
        return "VOLATILE", 0.4
    
    def _get_recommended_strategy(
        self,
        regime: str,
        volatility_state: str,
        confidence: float
    ) -> str:
        """Get recommended trading strategy for current regime."""
        if confidence < 0.3:
            return "STAY_OUT"
        
        if regime in ["TRENDING_UP", "TRENDING_DOWN"]:
            return "TREND_FOLLOW"
        elif regime == "RANGING":
            return "MEAN_REVERT"
        elif regime == "VOLATILE":
            if volatility_state in ["EXTREME", "HIGH"]:
                return "STAY_OUT"
            else:
                return "MEAN_REVERT"
        
        return "STAY_OUT"
    
    def get_strategy_adjustment(self, regime: RegimeState) -> Dict:
        """
        Return parameter adjustments for the current regime.
        """
        adjustments = {
            'regime': regime.regime,
            'confidence': regime.confidence,
            'position_size_multiplier': 1.0,
            'stop_loss_pips': 20,
            'take_profit_multiplier': 1.5,
            'signal_threshold': 0.5,
            'use_trailing_stop': False,
            'breakeven_at_pips': 10
        }
        
        if regime.recommended_strategy == "STAY_OUT":
            adjustments['position_size_multiplier'] = 0.0
            adjustments['signal_threshold'] = 1.0
            return adjustments
        
        if regime.regime == "TRENDING_UP" or regime.regime == "TRENDING_DOWN":
            adjustments['position_size_multiplier'] = 1.2 * regime.confidence
            adjustments['stop_loss_pips'] = 30
            adjustments['take_profit_multiplier'] = 2.0
            adjustments['signal_threshold'] = 0.4
            adjustments['use_trailing_stop'] = True
            adjustments['breakeven_at_pips'] = 15
            
            if regime.volatility_state == "EXTREME":
                adjustments['position_size_multiplier'] *= 0.6
                adjustments['stop_loss_pips'] = 40
        
        elif regime.regime == "RANGING":
            adjustments['position_size_multiplier'] = 1.1 * regime.confidence
            adjustments['stop_loss_pips'] = 15
            adjustments['take_profit_multiplier'] = 1.0
            adjustments['signal_threshold'] = 0.4
            adjustments['use_trailing_stop'] = False
            adjustments['breakeven_at_pips'] = 8
        
        elif regime.regime == "VOLATILE":
            if regime.volatility_state == "EXTREME":
                adjustments['position_size_multiplier'] = 0.4
                adjustments['signal_threshold'] = 0.7
                adjustments['stop_loss_pips'] = 50
            else:
                adjustments['position_size_multiplier'] = 0.7 * regime.confidence
                adjustments['stop_loss_pips'] = 35
            
            adjustments['take_profit_multiplier'] = 1.2
            adjustments['use_trailing_stop'] = False
        
        adjustments['position_size_multiplier'] *= (0.5 + 0.5 * regime.confidence)
        
        return adjustments
    
    def get_regime_history(self) -> List[Dict]:
        """Return history of regime transitions."""
        return self.regime_history.copy()
    
    def get_current_indicators(self) -> Dict:
        """Get current indicator values for debugging."""
        if len(self.closes) < self.adx_period:
            return {}
        
        adx = self._calculate_adx()
        plus_di, minus_di = self._calculate_di()
        atr = self._calculate_atr()[-1]
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands()
        ema21 = self._calculate_ema(self.ema21_period)
        ema50 = self._calculate_ema(self.ema50_period)
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'atr': atr,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'ema21': ema21,
            'ema50': ema50,
            'current_price': float(self.closes[-1]) if len(self.closes) > 0 else 0.0,
            'bars_processed': len(self.closes)
        }
