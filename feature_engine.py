"""
ML SuperTrend v51 - Auto Feature Engineering + Anomaly Detection
================================================================
Automatically generates derived features from raw indicators and
detects anomalous market conditions using Isolation Forest.

Features generated:
  - Cross-indicator interactions (RSI×ADX, MACD×Volume, etc.)
  - Lag features (change rates over 1, 3, 5 bars)
  - Statistical features (z-scores, percentile ranks)
  - Regime-conditional features

Anomaly detection:
  - Isolation Forest on multi-dimensional indicator space
  - Flags unusual market conditions (flash crashes, liquidity gaps, etc.)
  - Score: -1 (anomaly) to +1 (normal)

Usage:
    from feature_engine import FeatureEngineer, AnomalyDetector
    fe = FeatureEngineer()
    features = fe.generate(indicators_dict, candles)

    ad = AnomalyDetector()
    ad.fit(historical_features)
    is_anomaly, score = ad.check(current_features)
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Try sklearn for Isolation Forest
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available \u2014 anomaly detection disabled")


class FeatureEngineer:
    """
    Automatically generates derived features from raw indicators.

    Creates interaction terms, lag features, z-scores, and
    regime-conditional features for ML model consumption.
    """

    def __init__(self, lookback_bars: int = 50):
        self.lookback_bars = lookback_bars
        # History buffers for lag features
        self._history: Dict[str, List[float]] = {}
        self._max_history = 100

    def generate(self, indicators: Dict, candles: List[Dict] = None) -> Dict[str, float]:
        """
        Generate engineered features from raw indicators.

        Args:
            indicators: Dict from compute_all_indicators()
            candles: Raw candle list (optional, for extra features)

        Returns:
            Dict of feature_name -> float value
        """
        features = {}

        # Extract raw values
        close = float(indicators.get('close', np.zeros(1))[-1]) if isinstance(indicators.get('close'), np.ndarray) else 0
        rsi = float(indicators.get('rsi', np.array([50]))[-1]) if isinstance(indicators.get('rsi'), np.ndarray) else 50
        adx = float(indicators.get('adx', np.array([20]))[-1]) if isinstance(indicators.get('adx'), np.ndarray) else 20
        atr = float(indicators.get('atr', np.array([0]))[-1]) if isinstance(indicators.get('atr'), np.ndarray) else 0
        macd_hist = float(indicators.get('macd_histogram', np.array([0]))[-1]) if isinstance(indicators.get('macd_histogram'), np.ndarray) else 0
        bb_width = float(indicators.get('bb_width', np.array([0]))[-1]) if isinstance(indicators.get('bb_width'), np.ndarray) else 0
        bb_pct_b = float(indicators.get('bb_pct_b', np.array([0.5]))[-1]) if isinstance(indicators.get('bb_pct_b'), np.ndarray) else 0.5
        vol_ratio = float(indicators.get('vol_ratio', np.array([1]))[-1]) if isinstance(indicators.get('vol_ratio'), np.ndarray) else 1.0
        ema_fast = float(indicators.get('ema_fast', np.array([0]))[-1]) if isinstance(indicators.get('ema_fast'), np.ndarray) else 0
        ema_slow = float(indicators.get('ema_slow', np.array([0]))[-1]) if isinstance(indicators.get('ema_slow'), np.ndarray) else 0

        # === 1. INTERACTION FEATURES ===
        # RSI × ADX: Strong trend + overbought/oversold = powerful signal
        features['rsi_x_adx'] = (rsi / 100.0) * (adx / 50.0)

        # MACD × Volume: Momentum with volume confirmation
        features['macd_x_vol'] = macd_hist * vol_ratio

        # BB position × RSI: Bollinger position confirmed by momentum
        features['bb_x_rsi'] = bb_pct_b * (rsi / 100.0)

        # EMA spread: How far apart fast and slow EMAs are (trend strength)
        if ema_slow != 0:
            features['ema_spread_pct'] = (ema_fast - ema_slow) / ema_slow * 100
        else:
            features['ema_spread_pct'] = 0

        # ATR × BB width: Volatility regime composite
        features['vol_composite'] = (atr * bb_width) if atr > 0 and bb_width > 0 else 0

        # === 2. Z-SCORE FEATURES ===
        # How far current values are from their recent means (in std devs)
        rsi_arr = indicators.get('rsi', np.array([50]))
        adx_arr = indicators.get('adx', np.array([20]))
        atr_arr = indicators.get('atr', np.array([0]))

        features['rsi_zscore'] = self._zscore(rsi_arr, 20)
        features['adx_zscore'] = self._zscore(adx_arr, 20)
        features['atr_zscore'] = self._zscore(atr_arr, 20)
        features['vol_zscore'] = self._zscore(indicators.get('vol_ratio', np.array([1])), 20)

        # === 3. RATE OF CHANGE FEATURES ===
        closes = indicators.get('close', np.array([0]))
        if len(closes) >= 5:
            features['roc_1'] = (closes[-1] / closes[-2] - 1) * 100 if closes[-2] != 0 else 0
            features['roc_3'] = (closes[-1] / closes[-4] - 1) * 100 if closes[-4] != 0 else 0
            features['roc_5'] = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 and closes[-6] != 0 else 0
        else:
            features['roc_1'] = features['roc_3'] = features['roc_5'] = 0

        # RSI rate of change
        if len(rsi_arr) >= 3:
            features['rsi_roc'] = float(rsi_arr[-1] - rsi_arr[-3])
        else:
            features['rsi_roc'] = 0

        # ADX rate of change (trend strengthening/weakening)
        if len(adx_arr) >= 5:
            features['adx_roc'] = float(adx_arr[-1] - adx_arr[-5])
        else:
            features['adx_roc'] = 0

        # === 4. PERCENTILE RANK FEATURES ===
        features['atr_percentile'] = self._percentile_rank(atr_arr, 50)
        features['vol_percentile'] = self._percentile_rank(indicators.get('vol_ratio', np.array([1])), 50)
        features['rsi_percentile'] = self._percentile_rank(rsi_arr, 50)

        # === 5. CANDLE PATTERN FEATURES ===
        if candles and len(candles) >= 3:
            last = candles[-1]
            prev = candles[-2]
            body = abs(last['close'] - last.get('open', last['close']))
            total_range = last['high'] - last['low']

            # Body ratio: how much of the candle is body vs wick
            features['body_ratio'] = body / total_range if total_range > 0 else 0.5

            # Upper/lower wick ratio
            if last['close'] >= last.get('open', last['close']):
                upper_wick = last['high'] - last['close']
                lower_wick = last.get('open', last['close']) - last['low']
            else:
                upper_wick = last['high'] - last.get('open', last['close'])
                lower_wick = last['close'] - last['low']
            features['upper_wick_pct'] = upper_wick / total_range if total_range > 0 else 0
            features['lower_wick_pct'] = lower_wick / total_range if total_range > 0 else 0

            # Consecutive direction count
            direction_count = 0
            for i in range(len(candles) - 1, max(len(candles) - 6, 0), -1):
                c = candles[i]
                is_bullish = c['close'] >= c.get('open', c['close'])
                if i == len(candles) - 1:
                    current_dir = is_bullish
                if is_bullish == current_dir:
                    direction_count += 1
                else:
                    break
            features['consec_candles'] = direction_count * (1 if current_dir else -1)
        else:
            features['body_ratio'] = 0.5
            features['upper_wick_pct'] = 0.25
            features['lower_wick_pct'] = 0.25
            features['consec_candles'] = 0

        # === 6. TIME FEATURES ===
        now = datetime.now(timezone.utc)
        features['hour_sin'] = np.sin(2 * np.pi * now.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * now.hour / 24)
        features['dow_sin'] = np.sin(2 * np.pi * now.weekday() / 7)
        features['dow_cos'] = np.cos(2 * np.pi * now.weekday() / 7)

        return features

    def get_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to ordered numpy array."""
        sorted_keys = sorted(features.keys())
        return np.array([features[k] for k in sorted_keys], dtype=np.float64)

    def get_feature_names(self, features: Dict[str, float]) -> List[str]:
        """Get ordered feature names matching the vector."""
        return sorted(features.keys())

    def _zscore(self, arr: np.ndarray, window: int = 20) -> float:
        """Calculate z-score of the last value over a rolling window."""
        if len(arr) < window:
            return 0.0
        window_data = arr[-window:]
        mean = np.mean(window_data)
        std = np.std(window_data)
        if std == 0:
            return 0.0
        return float((arr[-1] - mean) / std)

    def _percentile_rank(self, arr: np.ndarray, window: int = 50) -> float:
        """Percentile rank of last value within window (0-1)."""
        if len(arr) < 5:
            return 0.5
        w = min(window, len(arr))
        window_data = arr[-w:]
        rank = np.sum(window_data <= arr[-1]) / len(window_data)
        return float(rank)


class AnomalyDetector:
    """
    Detects anomalous market conditions using Isolation Forest.

    Anomalies include:
    - Flash crashes / liquidity gaps
    - Unusual volatility spikes
    - Correlation breakdowns
    - Volume dry-ups before major moves

    When an anomaly is detected, the bot should either skip the trade
    or use ultra-conservative position sizing.
    """

    def __init__(
        self,
        contamination: float = 0.05,    # Expected fraction of anomalies
        n_estimators: int = 100,          # Number of trees
        min_samples_to_fit: int = 100,    # Minimum samples before fitting
        refit_interval: int = 200,        # Refit every N new samples
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.min_samples_to_fit = min_samples_to_fit
        self.refit_interval = refit_interval

        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.training_data: List[np.ndarray] = []
        self.samples_since_fit = 0
        self.feature_names: List[str] = []

        # Stats
        self.total_checks = 0
        self.total_anomalies = 0

        if not SKLEARN_AVAILABLE:
            logger.warning("AnomalyDetector: sklearn not available, will be a no-op")

    def add_sample(self, features: Dict[str, float]):
        """Add a new feature sample for training."""
        if not SKLEARN_AVAILABLE:
            return

        sorted_keys = sorted(features.keys())
        if not self.feature_names:
            self.feature_names = sorted_keys

        vector = np.array([features[k] for k in sorted_keys], dtype=np.float64)

        # Handle NaN/Inf
        if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

        self.training_data.append(vector)
        self.samples_since_fit += 1

        # Auto-fit when enough data
        if not self.is_fitted and len(self.training_data) >= self.min_samples_to_fit:
            self.fit()
        elif self.is_fitted and self.samples_since_fit >= self.refit_interval:
            self.fit()

    def fit(self):
        """Fit the Isolation Forest model on accumulated data."""
        if not SKLEARN_AVAILABLE or len(self.training_data) < self.min_samples_to_fit:
            return

        try:
            X = np.array(self.training_data[-1000:])  # Use last 1000 samples max

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            self.model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1,
            )
            self.model.fit(X_scaled)
            self.is_fitted = True
            self.samples_since_fit = 0

            logger.info(f"AnomalyDetector fitted on {len(X)} samples ({X.shape[1]} features)")
        except Exception as e:
            logger.error(f"AnomalyDetector fit failed: {e}")

    def check(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """
        Check if current market conditions are anomalous.

        Returns:
            (is_anomaly, anomaly_score)
            - is_anomaly: True if conditions are anomalous
            - anomaly_score: -1 (anomaly) to +1 (normal). Lower = more anomalous.
        """
        self.total_checks += 1

        if not SKLEARN_AVAILABLE or not self.is_fitted or self.model is None:
            return False, 0.0

        try:
            sorted_keys = sorted(features.keys())
            vector = np.array([features[k] for k in sorted_keys], dtype=np.float64).reshape(1, -1)

            # Handle NaN/Inf
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

            vector_scaled = self.scaler.transform(vector)

            # Predict: -1 = anomaly, 1 = normal
            prediction = self.model.predict(vector_scaled)[0]
            # Score: lower = more anomalous
            score = float(self.model.score_samples(vector_scaled)[0])

            is_anomaly = prediction == -1
            if is_anomaly:
                self.total_anomalies += 1
                logger.warning(f"ANOMALY detected: score={score:.4f}")

            return is_anomaly, score

        except Exception as e:
            logger.warning(f"Anomaly check failed: {e}")
            return False, 0.0

    def get_status(self) -> Dict:
        """Get anomaly detector status for dashboard."""
        return {
            "is_fitted": self.is_fitted,
            "training_samples": len(self.training_data),
            "total_checks": self.total_checks,
            "total_anomalies": self.total_anomalies,
            "anomaly_rate": self.total_anomalies / max(1, self.total_checks),
            "sklearn_available": SKLEARN_AVAILABLE,
        }
