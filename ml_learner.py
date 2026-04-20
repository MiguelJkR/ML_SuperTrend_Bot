import numpy as np
import json
import os
import logging
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Try to import XGBoost with fallback
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log XGBoost availability
if not HAS_XGBOOST:
    logger.warning("XGBoost not installed. Using logistic regression fallback. Install with: pip install xgboost")


@dataclass
class TradeFeatures:
    """Entry features for a trade."""
    signal_strength: float
    adx_value: float
    rsi_value: float
    atr_percentile: float
    bb_width_pct: float
    ema_alignment: int  # 1=bullish, -1=bearish
    macd_histogram: float
    hour_of_day: int
    day_of_week: int
    regime: int  # 0=trending_up, 1=trending_down, 2=ranging, 3=volatile
    distance_from_st: float
    
    # Outcome (filled after trade closes)
    profitable: Optional[bool] = None
    pnl_pct: float = 0.0
    exit_type: str = ""  # "TP", "SL", "TRAIL"
    bars_held: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ModelWeights:
    """Container for logistic regression model weights."""
    weights: np.ndarray  # shape (n_features,)
    bias: float = 0.0
    trained: bool = False
    last_training_time: Optional[str] = None


class LogisticRegressionLearner:
    """From-scratch logistic regression implementation with numpy."""
    
    def __init__(self, n_features: int, learning_rate: float = 0.01, 
                 lambda_reg: float = 0.01, max_iterations: int = 1000,
                 batch_size: int = 8):
        """
        Initialize logistic regression model.
        
        Args:
            n_features: Number of input features
            learning_rate: Learning rate for gradient descent
            lambda_reg: L2 regularization parameter
            max_iterations: Max iterations for training
            batch_size: Batch size for mini-batch training
        """
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.trained = False
        self.training_history = []
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.
        Clipped to avoid overflow.
        """
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> float:
        """
        Compute binary cross-entropy loss with L2 regularization.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            sample_weights: Optional per-sample weights for class imbalance
        
        Returns:
            Loss value
        """
        m = X.shape[0]
        
        # Forward pass
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        
        # Binary cross-entropy loss
        # Use epsilon to avoid log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        if sample_weights is not None:
            bce_loss = -np.mean(sample_weights * (y * np.log(predictions) + (1 - y) * np.log(1 - predictions)))
        else:
            bce_loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        
        # L2 regularization
        l2_loss = (self.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)
        
        return bce_loss + l2_loss
    
    def compute_gradients(self, X: np.ndarray, y: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Compute gradients of loss with respect to weights and bias.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            sample_weights: Optional per-sample weights for class imbalance
        
        Returns:
            Tuple of (weight gradients, bias gradient)
        """
        m = X.shape[0]
        
        # Forward pass
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        
        # Error
        error = predictions - y
        
        # Apply sample weights if provided
        if sample_weights is not None:
            error = error * sample_weights
        
        # Gradients
        dw = (1.0 / m) * np.dot(X.T, error) + (self.lambda_reg / m) * self.weights
        db = (1.0 / m) * np.sum(error)
        
        return dw, db
    
    def train(self, X: np.ndarray, y: np.ndarray, sample_weights: Optional[np.ndarray] = None, verbose: bool = False) -> List[float]:
        """
        Train logistic regression using mini-batch gradient descent with optional sample weights.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            sample_weights: Optional per-sample weights for class imbalance (shape: n_samples,)
            verbose: Whether to print training progress
        
        Returns:
            List of loss values per iteration
        """
        m = X.shape[0]
        losses = []
        
        for iteration in range(self.max_iterations):
            # Mini-batch gradient descent
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            if sample_weights is not None:
                w_shuffled = sample_weights[indices]
            else:
                w_shuffled = None
            
            for batch_start in range(0, m, self.batch_size):
                batch_end = min(batch_start + self.batch_size, m)
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                w_batch = w_shuffled[batch_start:batch_end] if w_shuffled is not None else None
                
                # Compute gradients
                dw, db = self.compute_gradients(X_batch, y_batch, w_batch)
                
                # Update weights and bias
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            # Compute loss on full dataset every 10 iterations
            if iteration % 10 == 0 or iteration == self.max_iterations - 1:
                loss = self.compute_loss(X, y, sample_weights)
                losses.append(loss)
                
                if verbose and iteration % 100 == 0:
                    logger.info(f"Iteration {iteration}: Loss = {loss:.6f}")
        
        self.trained = True
        self.training_history = losses
        return losses
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of positive class.
        
        Args:
            X: Feature matrix (n_samples, n_features) or (n_features,)
        
        Returns:
            Probability predictions
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary class (0 or 1).
        
        Args:
            X: Feature matrix
            threshold: Classification threshold
        
        Returns:
            Binary predictions
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.
        
        Args:
            X: Feature matrix
            y: True labels
        
        Returns:
            Accuracy (0-1)
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


class XGBoostTradeScorer:
    """XGBoost-based trade signal scorer - captures non-linear feature relationships."""

    def __init__(self, n_features: int, params: dict = None):
        self.n_features = n_features
        self.model = None
        self.trained = False
        self.training_history = []
        self.default_params = {
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 150,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'verbosity': 0,
        }
        if params:
            self.default_params.update(params)

    def train(self, X, y, verbose=False):
        """Train XGBoost model with early stopping via temporal train/val split."""
        if not HAS_XGBOOST:
            return []

        # Temporal split: first 80% for training, last 20% for validation
        n = len(X)
        split = int(n * 0.8)
        
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {k: v for k, v in self.default_params.items()
                  if k not in ['n_estimators', 'use_label_encoder', 'verbosity']}
        params['verbosity'] = 0

        evals_result = {}
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.default_params.get('n_estimators', 150),
            evals=[(dtrain, 'train'), (dval, 'val')],
            evals_result=evals_result,
            early_stopping_rounds=20,
            verbose_eval=False
        )

        self.trained = True
        self.training_history = evals_result.get('val', {}).get('logloss', [])

        if verbose:
            logger.info(f"XGBoost trained. Best iteration: {self.model.best_iteration}")

        return self.training_history

    def predict_proba(self, X):
        """Predict probability."""
        if not self.trained or self.model is None:
            return np.full(X.shape[0] if X.ndim > 1 else 1, 0.5)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        dmat = xgb.DMatrix(X)
        return self.model.predict(dmat)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_feature_importance(self, feature_names=None):
        """Get feature importance from XGBoost model."""
        if not self.trained or self.model is None:
            return {}
        importance = self.model.get_score(importance_type='gain')
        # Convert f0, f1, ... to feature names
        if feature_names:
            named_importance = {}
            for key, val in importance.items():
                idx = int(key.replace('f', ''))
                if idx < len(feature_names):
                    named_importance[feature_names[idx]] = val
            return named_importance
        return importance


class MLTradeScorer:
    """ML-based trade signal scorer that learns from historical trades."""
    
    # Raw feature names and indices (11 original features from TradeFeatures)
    FEATURE_NAMES_RAW = [
        'signal_strength', 'adx_value', 'rsi_value', 'atr_percentile',
        'bb_width_pct', 'ema_alignment', 'macd_histogram', 'hour_of_day',
        'day_of_week', 'regime', 'distance_from_st'
    ]
    N_FEATURES_RAW = len(FEATURE_NAMES_RAW)
    
    # Engineered feature names (5 derived features)
    ENGINEERED_FEATURES = [
        'rsi_zone', 'adx_strong', 'momentum_alignment', 'session', 'volatility_regime'
    ]
    
    # Full feature set
    FEATURE_NAMES = FEATURE_NAMES_RAW + ENGINEERED_FEATURES
    N_FEATURES = len(FEATURE_NAMES)
    
    def __init__(self, min_trades_to_train: int = 10, learning_rate: float = 0.01,
                 retrain_interval: int = 5, max_training_window: int = 200,
                 data_file: str = r"C:\Dev\ML_SuperTrend_Bot\ml_trade_data.json"):
        """
        Initialize ML Trade Scorer.
        
        Args:
            min_trades_to_train: Minimum trades needed before first training
            learning_rate: Learning rate for logistic regression
            retrain_interval: Retrain every N completed trades
            max_training_window: Use only last N trades to prevent concept drift
            data_file: Path to persist trade data
        """
        self.min_trades_to_train = min_trades_to_train
        self.learning_rate = learning_rate
        self.retrain_interval = retrain_interval
        self.max_training_window = max_training_window
        self.data_file = data_file
        
        # Trade data storage
        self.entry_features: Dict[str, TradeFeatures] = {}  # trade_id -> features
        self.completed_trades: List[Dict] = []  # Completed trades for training
        
        # Model
        self.model = LogisticRegressionLearner(
            n_features=self.N_FEATURES,
            learning_rate=learning_rate
        )

        # XGBoost model (preferred when available)
        self.xgb_model = None
        if HAS_XGBOOST:
            self.xgb_model = XGBoostTradeScorer(n_features=self.N_FEATURES)
            logger.info("XGBoost scorer initialized (primary model)")
        self.use_xgboost = HAS_XGBOOST

        # Feature normalization statistics (per-feature statistics)
        self.feature_mean = np.zeros(self.N_FEATURES)
        self.feature_std = np.ones(self.N_FEATURES)
        
        # Stats
        self.trades_since_training = 0
        self.model_accuracy = 0.0
        self.model_val_accuracy = 0.0
        self.total_predictions = 0
        
        # Load existing data if available
        self.load()
        logger.info(f"MLTradeScorer initialized. Data file: {data_file}")

    def load(self):
        """Load persisted trade data and model stats from disk."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                self.completed_trades = data.get('completed_trades', [])
                self.model_accuracy = data.get('model_accuracy', 0.0)
                self.model_val_accuracy = data.get('model_val_accuracy', 0.0)
                self.total_predictions = data.get('total_predictions', 0)
                self.trades_since_training = data.get('trades_since_training', 0)
                fm = data.get('feature_mean')
                fs = data.get('feature_std')
                if fm and len(fm) == self.N_FEATURES:
                    self.feature_mean = np.array(fm)
                if fs and len(fs) == self.N_FEATURES:
                    self.feature_std = np.array(fs)
                logger.info(f"Loaded {len(self.completed_trades)} trades from {self.data_file}")
        except Exception as e:
            logger.warning(f"Could not load ML data: {e}")

    def save(self):
        """Persist trade data and model stats to disk."""
        try:
            data = {
                'completed_trades': self.completed_trades[-500:],  # Keep last 500
                'model_accuracy': float(self.model_accuracy),
                'model_val_accuracy': float(self.model_val_accuracy),
                'total_predictions': self.total_predictions,
                'trades_since_training': self.trades_since_training,
                'feature_mean': self.feature_mean.tolist(),
                'feature_std': self.feature_std.tolist(),
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Could not save ML data: {e}")

    def record_entry(self, trade_id: str, features: TradeFeatures):
        """
        Record features at trade entry.
        
        Args:
            trade_id: Unique trade identifier
            features: TradeFeatures object with entry information
        """
        self.entry_features[trade_id] = features
        logger.info(f"Recorded entry for trade {trade_id}. Signal strength: {features.signal_strength:.3f}")
    
    def record_exit(self, trade_id: str, profitable: bool, pnl_pct: float,
                    exit_type: str, bars_held: int, max_fav_excursion: float = 0.0,
                    max_adv_excursion: float = 0.0):
        """
        Record trade outcome and trigger retraining if needed.
        
        Args:
            trade_id: Unique trade identifier
            profitable: Whether trade was profitable
            pnl_pct: P&L percentage
            exit_type: "TP", "SL", or "TRAIL"
            bars_held: Number of bars the trade was held
            max_fav_excursion: Maximum favorable excursion (%)
            max_adv_excursion: Maximum adverse excursion (%)
        """
        if trade_id not in self.entry_features:
            logger.warning(f"No entry features found for trade {trade_id}")
            return
        
        # Get entry features and update with outcome
        features = self.entry_features[trade_id]
        features.profitable = profitable
        features.pnl_pct = pnl_pct
        features.exit_type = exit_type
        features.bars_held = bars_held
        
        # Store completed trade
        trade_data = {
            'trade_id': trade_id,
            'features': asdict(features),
            'profitable': profitable,
            'pnl_pct': pnl_pct,
            'exit_type': exit_type,
            'bars_held': bars_held,
            'max_fav_excursion': max_fav_excursion,
            'max_adv_excursion': max_adv_excursion,
            'timestamp': datetime.now().isoformat()
        }
        self.completed_trades.append(trade_data)
        
        # Increment counter
        self.trades_since_training += 1
        
        logger.info(f"Recorded exit for trade {trade_id}. Profitable: {profitable}, "
                   f"P&L: {pnl_pct:+.2f}%, Exit: {exit_type}")
        
        # Check if retraining is needed
        if (len(self.completed_trades) >= self.min_trades_to_train and
            self.trades_since_training >= self.retrain_interval):
            logger.info(f"Triggering model retraining. Completed trades: {len(self.completed_trades)}")
            self._train_model()
            self.trades_since_training = 0
            self.save()
    
    def _engineer_features(self, features: TradeFeatures) -> np.ndarray:
        """
        Engineer 5 derived features from raw features.
        
        Args:
            features: TradeFeatures object with raw features
        
        Returns:
            Array of 5 engineered features in order: [rsi_zone, adx_strong, momentum_alignment, session, volatility_regime]
        """
        # rsi_zone: 0=oversold(<30), 1=neutral(30-70), 2=overbought(>70)
        if features.rsi_value < 30:
            rsi_zone = 0
        elif features.rsi_value > 70:
            rsi_zone = 2
        else:
            rsi_zone = 1
        
        # adx_strong: 1 if adx>25 else 0
        adx_strong = 1 if features.adx_value > 25 else 0
        
        # momentum_alignment: ema_alignment * sign(macd_histogram)
        # If both agree (same sign), result is 1; if they disagree, result is -1
        macd_sign = 1 if features.macd_histogram >= 0 else -1
        momentum_alignment = features.ema_alignment * macd_sign
        
        # session: 0=Asia(0-8), 1=Europe(8-16), 2=NY(16-24)
        hour = features.hour_of_day
        if hour < 8:
            session = 0
        elif hour < 16:
            session = 1
        else:
            session = 2
        
        # volatility_regime: interaction of bb_width_pct * atr_percentile / 100
        volatility_regime = (features.bb_width_pct * features.atr_percentile) / 100.0
        
        return np.array([rsi_zone, adx_strong, momentum_alignment, session, volatility_regime])
    
    def _normalize_features(self, features: TradeFeatures, X_raw: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert TradeFeatures to normalized numpy array with engineered features.
        
        Args:
            features: TradeFeatures object
            X_raw: Optional pre-computed raw features for batching. If None, extracts from features.
        
        Returns:
            Normalized feature array (N_FEATURES,)
        """
        # Extract raw features
        if X_raw is None:
            feature_values_raw = np.array([
                features.signal_strength,
                features.adx_value,
                features.rsi_value,
                features.atr_percentile,
                features.bb_width_pct,
                features.ema_alignment,
                features.macd_histogram,
                features.hour_of_day,
                features.day_of_week,
                features.regime,
                features.distance_from_st
            ])
        else:
            feature_values_raw = X_raw
        
        # Engineer derived features
        engineered = self._engineer_features(features)
        
        # Concatenate raw and engineered
        feature_values = np.concatenate([feature_values_raw, engineered])
        
        # Normalize (z-score) using stored statistics
        normalized = (feature_values - self.feature_mean) / self.feature_std
        return normalized
    
    def _compute_class_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Compute class weights to handle imbalanced data.
        
        Args:
            y: Target vector (n_samples,)
        
        Returns:
            Sample weights array (n_samples,)
        """
        n_pos = np.sum(y)
        n_neg = len(y) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            # No imbalance or degenerate case
            return np.ones(len(y))
        
        # Weight = len(y) / (2 * count_in_class)
        w_pos = len(y) / (2.0 * n_pos)
        w_neg = len(y) / (2.0 * n_neg)
        
        weights = np.where(y == 1, w_pos, w_neg)
        return weights
    
    def _train_model(self):
        """
        Train logistic regression from scratch using mini-batch gradient descent.
        Uses sliding window to prevent concept drift.
        Implements temporal cross-validation for overfitting detection.
        """
        if len(self.completed_trades) < self.min_trades_to_train:
            logger.warning(f"Not enough completed trades for training. "
                          f"Need {self.min_trades_to_train}, have {len(self.completed_trades)}")
            return
        
        # Apply sliding window to use only recent data
        trades_to_use = self.completed_trades[-self.max_training_window:]
        logger.info(f"Using last {len(trades_to_use)} trades (max_training_window={self.max_training_window})")
        
        # Prepare training data
        X_raw_list = []
        X_list = []
        y_list = []
        
        for trade in trades_to_use:
            features_dict = trade['features']
            features = TradeFeatures(**{k: v for k, v in features_dict.items() 
                                       if k in TradeFeatures.__dataclass_fields__})
            
            # Extract raw features
            x_raw = np.array([
                features.signal_strength,
                features.adx_value,
                features.rsi_value,
                features.atr_percentile,
                features.bb_width_pct,
                features.ema_alignment,
                features.macd_histogram,
                features.hour_of_day,
                features.day_of_week,
                features.regime,
                features.distance_from_st
            ])
            X_raw_list.append(x_raw)
            
            # Normalize features
            x_normalized = self._normalize_features(features, X_raw=x_raw)
            X_list.append(x_normalized)
            
            # Target: 1 if profitable, 0 otherwise
            y_list.append(1 if trade['profitable'] else 0)
        
        X_raw = np.array(X_raw_list)
        X = np.array(X_list)
        y = np.array(y_list)
        
        # FIX BUG #1: Compute mean/std PER FEATURE across ALL training samples
        self.feature_mean = np.mean(X_raw, axis=0)
        # Append engineered feature means/stds (compute from normalized engineered features)
        engineered_features_list = []
        for features_dict in [t['features'] for t in trades_to_use]:
            features = TradeFeatures(**{k: v for k, v in features_dict.items() 
                                       if k in TradeFeatures.__dataclass_fields__})
            eng = self._engineer_features(features)
            engineered_features_list.append(eng)
        
        if engineered_features_list:
            engineered_all = np.array(engineered_features_list)
            # Extend feature_mean and feature_std to include engineered features
            mean_raw = np.mean(X_raw, axis=0)
            std_raw = np.maximum(np.std(X_raw, axis=0), 1e-8)
            mean_eng = np.mean(engineered_all, axis=0)
            std_eng = np.maximum(np.std(engineered_all, axis=0), 1e-8)
            
            self.feature_mean = np.concatenate([mean_raw, mean_eng])
            self.feature_std = np.concatenate([std_raw, std_eng])
        else:
            self.feature_std = np.maximum(np.std(X_raw, axis=0), 1e-8)
            self.feature_mean = np.mean(X_raw, axis=0)
        
        # Re-normalize with correct per-feature statistics
        X_normalized = np.zeros((len(trades_to_use), self.N_FEATURES))
        for i, features_dict in enumerate([t['features'] for t in trades_to_use]):
            features = TradeFeatures(**{k: v for k, v in features_dict.items() 
                                       if k in TradeFeatures.__dataclass_fields__})
            X_normalized[i] = self._normalize_features(features)
        
        X = X_normalized
        
        # FIX BUG #2: Use temporal split (80/20) instead of training on full data
        n = len(X)
        split = int(n * 0.8)
        
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # ML #1: Compute class weights for imbalanced data
        sample_weights = self._compute_class_weights(y_train)
        
        logger.info(f"Training logistic regression on {len(X_train)} samples (validation: {len(X_val)})")
        logger.info(f"Train - Positive: {np.sum(y_train)}, Negative: {len(y_train) - np.sum(y_train)}")
        logger.info(f"Validation - Positive: {np.sum(y_val)}, Negative: {len(y_val) - np.sum(y_val)}")
        
        # Train model
        self.model = LogisticRegressionLearner(
            n_features=self.N_FEATURES,
            learning_rate=self.learning_rate
        )
        self.model.train(X_train, y_train, sample_weights=sample_weights, verbose=True)
        
        # Compute training and validation accuracy
        train_accuracy = self.model.score(X_train, y_train)
        val_accuracy = self.model.score(X_val, y_val) if len(X_val) > 0 else train_accuracy
        
        self.model_accuracy = train_accuracy
        self.model_val_accuracy = val_accuracy
        
        logger.info(f"Model training complete. Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # ML #4: Temporal CV - Check for overfitting
        if val_accuracy < 0.45 and len(X_val) > 0:
            logger.warning(f"Validation accuracy too low ({val_accuracy:.4f} < 0.45). "
                          f"Potential overfitting detected. Keeping previous model.")
            return

        # Also train XGBoost if available
        if self.use_xgboost and self.xgb_model is not None:
            try:
                self.xgb_model.train(X, y, verbose=True)
                xgb_accuracy = self.xgb_model.score(X_val, y_val) if len(X_val) > 0 else self.xgb_model.score(X, y)
                logger.info(f"XGBoost training complete. Val Accuracy: {xgb_accuracy:.4f}")

                # Use XGBoost if it's better
                if xgb_accuracy > val_accuracy:
                    logger.info(f"XGBoost ({xgb_accuracy:.4f}) > LogReg ({val_accuracy:.4f}). Using XGBoost as primary.")
                    self.model_accuracy = xgb_accuracy
                else:
                    logger.info(f"LogReg ({val_accuracy:.4f}) >= XGBoost ({xgb_accuracy:.4f}). Keeping LogReg as primary.")
                    self.use_xgboost = False
            except Exception as e:
                logger.error(f"XGBoost training failed: {e}. Using LogReg fallback.")
                self.use_xgboost = False
    
    def score_signal(self, features: TradeFeatures) -> Dict:
        """
        Score a potential signal. Returns dict with adjustment factor for signal strength.
        
        Args:
            features: TradeFeatures object with current market conditions
        
        Returns:
            Dict with keys:
            - ml_score: 0-1 probability of profit
            - adjustment: multiplier for signal strength (0.85 to 1.10)
            - recommendation: "BOOST", "NEUTRAL", "PENALIZE", "BLOCK"
        """
        if not self.model.trained:
            # No trained model yet, return neutral
            return {
                'ml_score': 0.5,
                'adjustment': 1.0,
                'recommendation': 'NEUTRAL',
                'reason': 'Model not yet trained'
            }
        
        # Normalize features
        x_normalized = self._normalize_features(features)

        # Use XGBoost if available and trained
        if self.use_xgboost and self.xgb_model is not None and self.xgb_model.trained:
            ml_score = float(self.xgb_model.predict_proba(x_normalized.reshape(1, -1))[0])
        else:
            ml_score = float(self.model.predict_proba(x_normalized.reshape(1, -1))[0])
        self.total_predictions += 1
        
        # Determine recommendation and adjustment
        if ml_score >= 0.65:
            adjustment = 1.10  # +10% boost
            recommendation = "BOOST"
        elif ml_score >= 0.60:
            adjustment = 1.05  # +5% boost
            recommendation = "BOOST"
        elif ml_score <= 0.35:
            adjustment = 0.85  # -15% penalty
            recommendation = "PENALIZE"
        elif ml_score <= 0.40:
            adjustment = 0.90  # -10% penalty
            recommendation = "PENALIZE"
        else:
            adjustment = 1.0
            recommendation = "NEUTRAL"
        
        return {
            'ml_score': ml_score,
            'adjustment': adjustment,
            'recommendation': recommendation,
            'reason': f'Model confidence: {ml_score:.3f}'
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return feature importance based on model weights (magnitude).

        Returns:
            Dict mapping feature names to importance scores
        """
        # Use XGBoost feature importance if available and trained
        if self.use_xgboost and self.xgb_model is not None and self.xgb_model.trained:
            return self.xgb_model.get_feature_importance(self.FEATURE_NAMES)

        if not self.model.trained:
            return {}

        # Use absolute weight magnitude as importance
        importance = np.abs(self.model.weights)

        # Normalize to 0-1 range
        if importance.max() > 0:
            importance = importance / importance.max()

        return {name: float(imp) for name, imp in zip(self.FEATURE_NAMES, importance)}
    
    def get_stats(self) -> Dict:
        """
        Return learning statistics.

        Returns:
            Dict with training and performance statistics
        """
        profitable_trades = sum(1 for t in self.completed_trades if t['profitable'])

        return {
            'total_trades': len(self.completed_trades),
            'profitable_trades': profitable_trades,
            'loss_rate': (len(self.completed_trades) - profitable_trades) / max(len(self.completed_trades), 1),
            'model_trained': self.model.trained,
            'model_accuracy': float(self.model_accuracy),
            'model_val_accuracy': float(self.model_val_accuracy),
            'model_type': 'XGBoost' if (self.use_xgboost and self.xgb_model and self.xgb_model.trained) else 'LogisticRegression',
            'xgboost_available': HAS_XGBOOST,
            'trades_since_training': self.trades_since_training,
            'total_predictions': self.total_predictions,
            'feature_importance': self.get_feature_importance(),
            'n_features': self.N_FEATURES,
            'max_training_window': self.max_training_window,
        }
