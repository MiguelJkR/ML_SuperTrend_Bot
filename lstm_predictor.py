"""
ML SuperTrend v51 - LSTM Directional Predictor with Multi-Head Temporal Attention
==================================================================================
GPU-Optimized architecture for RTX 4070 Super + i7-14700KF.

Architecture v4 — GPU Multi-Head Attention + Advanced Learning:
  Input : Sequence of [close, rsi, adx, macd, atr, volume] + embeddings
          over last 30 bars (configurable)
  RNN   : LSTM 128 hidden units, 2 layers with dropout
  Attn  : Multi-Head Temporal Attention (4 heads, d_k=32 per head)
          Each head learns DIFFERENT temporal patterns
  Head  : AttentionPooled(H) → Dense 64 → ReLU → Dropout → Dense 32 → ReLU → Dense 1 → Sigmoid
  Quant : Optional Quantile Head → predicts [Q10, Q25, Q50, Q75, Q90]
  Output: Probability of UP move (0.0-1.0) + per-head attention weights + uncertainty

Advanced Learning (v4):
  - Sharpe-Aware Loss: optimizes risk-adjusted returns directly
  - MC Dropout: Bayesian uncertainty estimation (N=20 forward passes)
  - Quantile Regression: full return distribution with Pinball Loss
  - Data Augmentation: 6 techniques (jitter, warp, slice, permute, mixup, scale)
  - Curriculum Learning: easy→hard progressive training
  - Financial Positional Encoding: temporal market context

Training (GPU):
  - Mini-batch training with DataLoader (batch_size=64)
  - Gradient clipping (max_norm=1.0)
  - OneCycleLR scheduler for fast convergence
  - Mixed precision (FP16) on supported GPUs
  - Trains on rolling windows of the last 2000 candles

Fallback:
  - If no GPU → PyTorch CPU mode (slower but same architecture)
  - If no PyTorch → NumpyGRU + single-head attention (CPU)

Usage:
    from lstm_predictor import LSTMPredictor
    pred = LSTMPredictor()
    pred.update(candles, indicators)
    prob_up, confidence = pred.predict()
    weights = pred.last_attention_weights
    uncertainty = pred.last_uncertainty  # MC Dropout
"""

import logging
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Try PyTorch first, then fall back to numpy implementation
TORCH_AVAILABLE = False
CUDA_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_VRAM = torch.cuda.get_device_properties(0).total_mem / 1024**3
        logger.info(f"GPU detectada: {GPU_NAME} ({GPU_VRAM:.1f} GB VRAM)")
    else:
        logger.info("PyTorch disponible pero sin CUDA — usando CPU")
except ImportError:
    logger.info("PyTorch no disponible — usando numpy GRU fallback")

# Advanced learning modules
try:
    from advanced_learning import (
        SharpeLoss, QuantileHead, MCDropoutPredictor,
        CurriculumScheduler, FinancialPositionalEncoding,
    )
    ADVANCED_LEARNING = True
except ImportError:
    ADVANCED_LEARNING = False
    logger.info("Advanced learning modules not available")

try:
    from data_augmentation import TimeSeriesAugmentor
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    logger.info("Data augmentation not available")

# v8: Wavelet + Deep models
try:
    from wavelet_features import WaveletDecomposer
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False
    logger.info("WaveletDecomposer not available")

try:
    from causal_features import WassersteinDriftDetector
    WASSERSTEIN_AVAILABLE = True
except ImportError:
    WASSERSTEIN_AVAILABLE = False
    logger.info("WassersteinDriftDetector not available")

try:
    from deep_models import ContrastiveLearner, MarketVAE, VariableSelectionNetwork
    DEEP_MODELS_AVAILABLE = True
except ImportError:
    DEEP_MODELS_AVAILABLE = False
    logger.info("Deep models (VSN/Contrastive/VAE) not available")


# =====================================================================
# EMBEDDING TABLES (shared by both backends)
# =====================================================================
INSTRUMENT_IDS = {
    "EUR_USD": 0, "GBP_USD": 1, "USD_JPY": 2, "USD_CHF": 3,
    "AUD_USD": 4, "NZD_USD": 5, "USD_CAD": 6, "XAU_USD": 7,
    "BTC_USDT": 8, "ETH_USDT": 9,
}
SESSION_IDS = {
    "ASIAN": 0, "LONDON": 1, "NEW_YORK": 2,
    "LONDON_ASIAN": 3, "LONDON_NY": 4,
    "OFF_HOURS": 5, "WEEKEND": 6,
}
EMBED_INSTRUMENT_DIM = 4
EMBED_SESSION_DIM = 3

# GPU Configuration
GPU_CONFIG = {
    "hidden_size": 128,          # 128h (was 32)
    "num_layers": 2,             # 2-layer LSTM
    "num_heads": 4,              # 4-head Multi-Head Attention
    "dropout": 0.2,
    "batch_size": 64,
    "epochs_per_train": 20,      # More epochs with GPU speed
    "lr": 0.001,
    "lr_min": 1e-6,
    "max_grad_norm": 1.0,        # Gradient clipping
    "weight_decay": 1e-4,        # L2 regularization
    "sequence_length": 30,       # Longer lookback with GPU
    "buffer_size": 2000,         # More data
    "use_amp": True,             # Mixed precision (FP16)
}

# CPU fallback config (lighter)
CPU_CONFIG = {
    "hidden_size": 32,
    "num_layers": 1,
    "num_heads": 1,
    "dropout": 0.1,
    "batch_size": 32,
    "epochs_per_train": 10,
    "lr": 0.001,
    "lr_min": 1e-5,
    "max_grad_norm": 1.0,
    "weight_decay": 0,
    "sequence_length": 20,
    "buffer_size": 1000,
    "use_amp": False,
}


class NumpyGRU:
    """
    GRU + Temporal Self-Attention in pure numpy.
    Implements: Attention(Q,K,V) = softmax(QK^T / √d_k) V
    Used as fallback when PyTorch is not available.
    """

    def __init__(self, input_size: int = 6, hidden_size: int = 16, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale = 0.1

        self.hidden_size = hidden_size
        self.input_size = input_size

        # GRU gates: Update (z), Reset (r), New (n)
        self.Wz = rng.randn(hidden_size, input_size + hidden_size) * scale
        self.bz = np.zeros(hidden_size)
        self.Wr = rng.randn(hidden_size, input_size + hidden_size) * scale
        self.br = np.zeros(hidden_size)
        self.Wn = rng.randn(hidden_size, input_size + hidden_size) * scale
        self.bn = np.zeros(hidden_size)

        # ── Temporal Self-Attention weights (Q, K, V projections) ──
        self.d_k = hidden_size
        self.Wq = rng.randn(hidden_size, hidden_size) * scale
        self.Wk = rng.randn(hidden_size, hidden_size) * scale
        self.Wv = rng.randn(hidden_size, hidden_size) * scale

        # Output layer
        self.Wo = rng.randn(1, hidden_size) * scale
        self.bo = np.zeros(1)

        # Embedding tables
        self.emb_instrument = rng.randn(len(INSTRUMENT_IDS), EMBED_INSTRUMENT_DIM) * 0.1
        self.emb_session = rng.randn(len(SESSION_IDS), EMBED_SESSION_DIM) * 0.1

        self.trained = False
        self.lr = 0.001
        self.last_attention_weights = None

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def _tanh(self, x):
        return np.tanh(np.clip(x, -10, 10))

    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / (e.sum() + 1e-10)

    def _gru_all_hidden(self, sequence: np.ndarray) -> np.ndarray:
        """Run GRU and collect ALL hidden states H = (seq_len, hidden_size)."""
        seq_len = len(sequence)
        H = np.zeros((seq_len, self.hidden_size))
        h = np.zeros(self.hidden_size)

        for t in range(seq_len):
            x = sequence[t]
            xh = np.concatenate([x, h])
            z = self._sigmoid(self.Wz @ xh + self.bz)
            r = self._sigmoid(self.Wr @ xh + self.br)
            xh_r = np.concatenate([x, r * h])
            n = self._tanh(self.Wn @ xh_r + self.bn)
            h = (1 - z) * h + z * n
            H[t] = h

        return H

    def _attention(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Temporal Self-Attention: Attention(Q,K,V) = softmax(QK^T / √d_k) V
        """
        q = H[-1] @ self.Wq.T
        K = H @ self.Wk.T
        V = H @ self.Wv.T
        scores = K @ q / np.sqrt(self.d_k)
        weights = self._softmax(scores)
        self.last_attention_weights = weights
        context = weights @ V
        return context, weights

    def forward(self, sequence: np.ndarray) -> float:
        """Forward: GRU → Attention → Output."""
        H = self._gru_all_hidden(sequence)
        context, _ = self._attention(H)
        logit = self.Wo @ context + self.bo
        return float(self._sigmoid(logit[0]))

    def train_batch(self, sequences: List[np.ndarray], labels: List[int], epochs: int = 5):
        """Train with attention-aware weight updates."""
        if len(sequences) < 10:
            return

        for epoch in range(epochs):
            total_loss = 0
            for seq, label in zip(sequences, labels):
                pred = self.forward(seq)
                error = label - pred
                total_loss += error ** 2

                H = self._gru_all_hidden(seq)
                context, weights = self._attention(H)

                self.Wo += self.lr * error * context.reshape(1, -1)
                self.bo += self.lr * error

                for t in range(len(seq)):
                    w = weights[t]
                    if w > 0.05:
                        grad_scale = self.lr * error * w * 0.1
                        self.Wq += grad_scale * np.outer(H[-1], H[t]) * 0.01
                        self.Wk += grad_scale * np.outer(H[t], H[-1]) * 0.01
                        self.Wv += grad_scale * np.outer(H[t], H[t]) * 0.01

            avg_loss = total_loss / len(sequences)
            if epoch == epochs - 1:
                logger.info(f"NumpyGRU+Attn epoch {epoch+1}: loss={avg_loss:.4f}")

        self.trained = True


if TORCH_AVAILABLE:
    class MultiHeadTemporalAttention(nn.Module):
        """
        Multi-Head Temporal Self-Attention (4 cabezas).
        Cada cabeza aprende patrones temporales DIFERENTES:
          - Head 0: tendencia reciente (últimos 5 bars)
          - Head 1: momentum medio (últimos 10-15 bars)
          - Head 2: soporte/resistencia (picos de atención dispersos)
          - Head 3: patrón cíclico (distribución periódica)

        Attention(Q,K,V) = softmax(QK^T / √d_k) V
        Multi-Head: Concat(head_1, ..., head_h) W^O
        """
        def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
            super().__init__()
            assert hidden_size % num_heads == 0, f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            self.num_heads = num_heads
            self.d_k = hidden_size // num_heads  # Per-head dimension
            self.hidden_size = hidden_size

            # Q, K, V projections for ALL heads (single matrix for efficiency)
            self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
            self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
            self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)

            # Output projection: concat heads → hidden_size
            self.W_o = nn.Linear(hidden_size, hidden_size, bias=False)

            self.dropout = nn.Dropout(dropout)
            self.last_weights = None  # (batch, num_heads, seq_len)

        def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            H: (batch, seq_len, hidden_size)
            Returns: (context, weights)
                context: (batch, hidden_size)
                weights: (batch, seq_len) — averaged across heads
            """
            batch_size, seq_len, _ = H.shape

            # Use last hidden state as query
            query_input = H[:, -1:, :]  # (batch, 1, hidden_size)

            # Project Q from last state, K and V from all states
            Q = self.W_q(query_input)    # (batch, 1, hidden_size)
            K = self.W_k(H)              # (batch, seq_len, hidden_size)
            V = self.W_v(H)              # (batch, seq_len, hidden_size)

            # Reshape into multiple heads: (batch, num_heads, seq/1, d_k)
            Q = Q.view(batch_size, 1, self.num_heads, self.d_k).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            # Q: (batch, num_heads, 1, d_k)
            # K: (batch, num_heads, seq_len, d_k)
            # V: (batch, num_heads, seq_len, d_k)

            # Scaled dot-product attention per head
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
            # scores: (batch, num_heads, 1, seq_len)

            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            # weights: (batch, num_heads, 1, seq_len)

            # Store per-head weights for visualization
            self.last_weights = weights.detach()  # (batch, num_heads, 1, seq_len)

            # Weighted sum
            context = torch.matmul(weights, V)  # (batch, num_heads, 1, d_k)
            context = context.squeeze(2)         # (batch, num_heads, d_k)

            # Concatenate heads
            context = context.reshape(batch_size, self.hidden_size)  # (batch, hidden_size)
            context = self.W_o(context)  # Final projection

            # Average weights across heads for summary
            avg_weights = weights.squeeze(2).mean(dim=1)  # (batch, seq_len)

            return context, avg_weights


    class TorchLSTMGPU(nn.Module):
        """
        PyTorch LSTM + Multi-Head Temporal Attention.
        GPU-optimized for RTX 4070 Super.

        Architecture:
          LSTM(128h, 2 layers, dropout=0.2)
          → Multi-Head Attention(4 heads, d_k=32)
          → FC(128→64) → ReLU → Dropout
          → FC(64→32) → ReLU
          → FC(32→1) → Sigmoid
        """

        def __init__(self, input_size: int = 6, config: dict = None):
            super().__init__()
            cfg = config or (GPU_CONFIG if CUDA_AVAILABLE else CPU_CONFIG)

            self.hidden_size = cfg["hidden_size"]
            self.num_layers = cfg["num_layers"]
            self.num_heads = cfg["num_heads"]

            # LSTM backbone
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=cfg["dropout"] if self.num_layers > 1 else 0,
            )

            # Multi-Head Temporal Attention
            self.attention = MultiHeadTemporalAttention(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                dropout=cfg["dropout"],
            )

            # Layer normalization after attention
            self.layer_norm = nn.LayerNorm(self.hidden_size)

            # Deeper classification head
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(cfg["dropout"]),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

            # Quantile regression head (optional, for return distribution)
            self.quantile_head = None
            if ADVANCED_LEARNING:
                try:
                    self.quantile_head = QuantileHead(
                        input_size=self.hidden_size,
                        quantiles=[0.10, 0.25, 0.50, 0.75, 0.90]
                    )
                except Exception:
                    pass

            # Embedding tables
            self.emb_instrument = nn.Embedding(len(INSTRUMENT_IDS), EMBED_INSTRUMENT_DIM)
            self.emb_session = nn.Embedding(len(SESSION_IDS), EMBED_SESSION_DIM)

            # Initialize weights
            self._init_weights()

        def _init_weights(self):
            """Xavier initialization for stable training."""
            for name, param in self.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

        def forward(self, x, instrument_id=None, session_id=None, return_context=False):
            """
            x: (batch, seq_len, input_size)
            Returns: (batch,) probabilities
            If return_context=True: also returns (batch, hidden_size) context vector
            """
            lstm_out, _ = self.lstm(x)                    # (batch, seq_len, hidden)
            context, attn_w = self.attention(lstm_out)     # (batch, hidden)
            context = self.layer_norm(context)              # Normalize
            out = self.classifier(context)                  # (batch, 1)
            prob = out.squeeze(-1)
            if return_context:
                return prob, context
            return prob

        def get_head_weights(self) -> Optional[np.ndarray]:
            """Return per-head attention weights (num_heads, seq_len)."""
            if self.attention.last_weights is None:
                return None
            # last_weights: (batch, num_heads, 1, seq_len)
            w = self.attention.last_weights[0, :, 0, :]  # (num_heads, seq_len)
            return w.cpu().numpy()


    # Keep backward-compatible alias
    class TemporalSelfAttention(nn.Module):
        """Single-head attention (backward compat). Delegates to MultiHeadTemporalAttention."""
        def __init__(self, hidden_size: int):
            super().__init__()
            self.mha = MultiHeadTemporalAttention(hidden_size, num_heads=1, dropout=0.0)
            self.last_weights = None

        def forward(self, H):
            context, weights = self.mha(H)
            self.last_weights = self.mha.last_weights
            return context, weights

    # Backward-compatible alias
    TorchLSTM = TorchLSTMGPU


class LSTMPredictor:
    """
    LSTM-based directional predictor for price movement.
    Auto-detects GPU and scales architecture accordingly.
    """

    def __init__(
        self,
        sequence_length: int = None,
        prediction_horizon: int = 5,
        hidden_size: int = None,
        min_samples_to_train: int = 100,
        retrain_interval: int = 50,
        features: List[str] = None,
        model_path: str = None,
    ):
        # Auto-select config based on hardware
        self.device = "cpu"
        if CUDA_AVAILABLE:
            self.config = GPU_CONFIG.copy()
            self.device = "cuda"
            logger.info(f"LSTMPredictor: Modo GPU ({GPU_NAME})")
        elif TORCH_AVAILABLE:
            self.config = CPU_CONFIG.copy()
            logger.info("LSTMPredictor: Modo CPU (PyTorch)")
        else:
            self.config = CPU_CONFIG.copy()
            logger.info("LSTMPredictor: Modo numpy fallback")

        # Allow overrides
        self.sequence_length = sequence_length or self.config["sequence_length"]
        self.hidden_size = hidden_size or self.config["hidden_size"]
        self.prediction_horizon = prediction_horizon
        self.min_samples_to_train = min_samples_to_train
        self.retrain_interval = retrain_interval
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "lstm_model.pt" if TORCH_AVAILABLE else "lstm_model.json"
        )

        self.features = features or ['close_norm', 'rsi_norm', 'adx_norm', 'macd_norm', 'atr_norm', 'vol_norm']
        self.input_size = len(self.features)

        # Data buffers
        self.data_buffer: List[Dict[str, float]] = []
        self.max_buffer = self.config["buffer_size"]

        # Model
        self.torch_model = None
        self.numpy_model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision
        self.is_trained = False
        self.samples_since_train = 0

        if TORCH_AVAILABLE:
            self.torch_model = TorchLSTMGPU(input_size=self.input_size, config=self.config)
            self.torch_model.to(self.device)

            self.optimizer = torch.optim.AdamW(
                self.torch_model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
            )

            # Mixed precision scaler for GPU
            if self.config["use_amp"] and CUDA_AVAILABLE:
                self.scaler = torch.amp.GradScaler('cuda')
                logger.info("Mixed precision (FP16) activado")

            param_count = sum(p.numel() for p in self.torch_model.parameters())
            logger.info(f"Modelo: {param_count:,} parametros | {self.hidden_size}h | "
                       f"{self.config['num_heads']} heads | device={self.device}")
        else:
            self.numpy_model = NumpyGRU(input_size=self.input_size, hidden_size=min(self.hidden_size, 16))
            logger.info("LSTMPredictor: Using numpy GRU fallback")

        # Stats
        self.total_predictions = 0
        self.correct_predictions = 0
        self.last_prediction = None
        self.last_actual = None
        self.last_attention_weights = None
        self.last_head_weights = None  # Per-head weights (num_heads, seq_len)
        self.current_instrument_id = 0
        self.current_session_id = 5
        self.last_train_loss = None
        self.train_losses = []  # History for dashboard loss curve
        self.gpu_stats = {}  # Temperature, VRAM, utilization

        # ===== Advanced Learning v4 =====
        self.mc_dropout = None
        self.sharpe_loss = None
        self.curriculum = None
        self.augmentor = None
        self.pos_encoding = None
        self.last_uncertainty = None   # MC Dropout result
        self.last_quantiles = None     # Quantile predictions

        if ADVANCED_LEARNING:
            try:
                self.mc_dropout = MCDropoutPredictor(
                    n_forward_passes=20,
                    uncertainty_threshold=0.15,
                )
                self.curriculum = CurriculumScheduler(
                    initial_pct=0.3,
                    growth_rate=0.1,
                )
                self.pos_encoding = FinancialPositionalEncoding(d_model=self.input_size)
                logger.info("Advanced learning: MC Dropout + Curriculum + Positional Encoding")
            except Exception as e:
                logger.warning(f"Advanced learning init error: {e}")

        if ADVANCED_LEARNING and TORCH_AVAILABLE:
            try:
                self.sharpe_loss = SharpeLoss(alpha=0.7)
                logger.info("Sharpe-Aware Loss initialized (α=0.7)")
            except Exception as e:
                logger.warning(f"SharpeLoss init error: {e}")

        if AUGMENTATION_AVAILABLE:
            try:
                self.augmentor = TimeSeriesAugmentor(
                    jitter_sigma=0.03,
                    scale_sigma=0.1,
                    warp_sigma=0.2,
                )
                logger.info("Data Augmentation initialized (6 techniques)")
            except Exception as e:
                logger.warning(f"Augmentor init error: {e}")

        # ===== v8: Wavelet + Deep Models =====
        self.wavelet = None
        self.wasserstein_drift = None
        self.contrastive_learner = None
        self.market_vae = None
        self.last_wavelet_energy = None
        self.last_drift_status = None
        self.last_vae_anomaly = None

        if WAVELET_AVAILABLE:
            try:
                self.wavelet = WaveletDecomposer(wavelet='db4', n_levels=4, denoise=True)
                logger.info("v8: WaveletDecomposer initialized (db4, 4 levels)")
            except Exception as e:
                logger.warning(f"WaveletDecomposer init error: {e}")

        if WASSERSTEIN_AVAILABLE:
            try:
                self.wasserstein_drift = WassersteinDriftDetector(
                    reference_window=200, test_window=50,
                    alert_threshold=0.5, critical_threshold=1.0
                )
                logger.info("v8: WassersteinDriftDetector initialized")
            except Exception as e:
                logger.warning(f"WassersteinDriftDetector init error: {e}")

        if DEEP_MODELS_AVAILABLE and TORCH_AVAILABLE:
            try:
                self.market_vae = MarketVAE(input_dim=self.input_size, latent_dim=8)
                self.contrastive_learner = ContrastiveLearner(
                    input_dim=self.input_size, hidden_dim=64, projection_dim=16
                )
                logger.info("v8: MarketVAE + ContrastiveLearner initialized")
            except Exception as e:
                logger.warning(f"Deep models init error: {e}")

    def update(self, candles: List[Dict], indicators: Dict):
        """Feed new market data. Triggers retraining when enough data arrives."""
        if not candles or not indicators:
            return

        try:
            closes = indicators.get('close', np.array([0]))
            rsi_arr = indicators.get('rsi', np.array([50]))
            adx_arr = indicators.get('adx', np.array([20]))
            macd_arr = indicators.get('macd_histogram', np.array([0]))
            atr_arr = indicators.get('atr', np.array([0]))
            vol_arr = indicators.get('vol_ratio', np.array([1]))

            close = float(closes[-1])
            close_mean = float(np.mean(closes[-50:])) if len(closes) >= 50 else close
            close_std = float(np.std(closes[-50:])) if len(closes) >= 50 else max(close * 0.01, 0.0001)

            snapshot = {
                'close_norm': (close - close_mean) / close_std if close_std > 0 else 0,
                'rsi_norm': float(rsi_arr[-1]) / 100.0,
                'adx_norm': min(float(adx_arr[-1]) / 50.0, 1.0),
                'macd_norm': np.clip(float(macd_arr[-1]) / (close * 0.001 + 0.0001), -1, 1),
                'atr_norm': min(float(atr_arr[-1]) / (close * 0.01 + 0.0001), 1.0),
                'vol_norm': min(float(vol_arr[-1]) / 3.0, 1.0),
                'close_raw': close,
            }

            self.data_buffer.append(snapshot)
            if len(self.data_buffer) > self.max_buffer:
                self.data_buffer = self.data_buffer[-self.max_buffer:]

            self.samples_since_train += 1

            # Check previous prediction accuracy
            if self.last_prediction is not None and len(self.data_buffer) >= 2:
                actual_up = self.data_buffer[-1]['close_raw'] > self.data_buffer[-2]['close_raw']
                predicted_up = self.last_prediction > 0.5
                if predicted_up == actual_up:
                    self.correct_predictions += 1

            # Auto-train
            if (not self.is_trained and len(self.data_buffer) >= self.min_samples_to_train) or \
               (self.is_trained and self.samples_since_train >= self.retrain_interval):
                self._train()

        except Exception as e:
            logger.warning(f"LSTM update error: {e}")

    def predict(self) -> Tuple[float, float]:
        """
        Predict probability of price going UP over next N bars.
        Uses MC Dropout for Bayesian uncertainty estimation when available.
        """
        if not self.is_trained or len(self.data_buffer) < self.sequence_length:
            return 0.5, 0.0

        try:
            sequence = self._build_sequence(self.data_buffer[-self.sequence_length:])

            if TORCH_AVAILABLE and self.torch_model:
                x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

                # MC Dropout: N forward passes con dropout activo → incertidumbre
                if self.mc_dropout and self.torch_model:
                    mc_result = self.mc_dropout.predict_with_uncertainty(self.torch_model, x)
                    prob_up = mc_result["mean"]
                    self.last_uncertainty = mc_result

                    # Si incertidumbre muy alta, reducir confianza
                    if not mc_result["should_trade"]:
                        logger.debug(f"MC Dropout: alta incertidumbre (std={mc_result['std']:.3f}), "
                                    f"reduciendo confianza")
                else:
                    self.torch_model.eval()
                    with torch.no_grad():
                        prob_up = float(self.torch_model(x).cpu().item())

                # Extract attention weights (always, for dashboard)
                self.torch_model.eval()
                with torch.no_grad():
                    # Run one clean forward pass for attention weights
                    _ = self.torch_model(x)
                    if self.torch_model.attention.last_weights is not None:
                        avg_w = self.torch_model.attention.last_weights[0, :, 0, :].mean(dim=0)
                        self.last_attention_weights = avg_w.cpu().numpy()
                        self.last_head_weights = self.torch_model.get_head_weights()

                # Quantile predictions (if available)
                if self.torch_model.quantile_head is not None:
                    try:
                        self.torch_model.eval()
                        with torch.no_grad():
                            _, context = self.torch_model(x, return_context=True)
                            q_preds = self.torch_model.quantile_head(context)
                            self.last_quantiles = self.torch_model.quantile_head.get_risk_metrics()
                    except Exception:
                        pass

            elif self.numpy_model:
                prob_up = self.numpy_model.forward(sequence)
                self.last_attention_weights = self.numpy_model.last_attention_weights
                self.last_head_weights = None
            else:
                return 0.5, 0.0

            # Confidence: base + MC Dropout adjustment
            confidence = abs(prob_up - 0.5) * 2
            if self.last_uncertainty and not self.last_uncertainty.get("should_trade", True):
                confidence *= 0.5  # Penalizar confianza si MC Dropout dice alta incertidumbre

            # v8: Wasserstein drift monitoring — reduce confidence during drift
            if self.wasserstein_drift and len(self.data_buffer) > 0:
                try:
                    last_close = self.data_buffer[-1].get('close_norm', 0)
                    self.wasserstein_drift.update(last_close)
                    drift_mult = self.wasserstein_drift.get_risk_multiplier()
                    self.last_drift_status = self.wasserstein_drift.get_status()
                    if drift_mult < 1.0:
                        confidence *= drift_mult
                        logger.debug(f"Wasserstein drift: {self.wasserstein_drift.alert_level}, "
                                    f"confidence adjusted by {drift_mult:.2f}")
                except Exception:
                    pass

            # v8: Wavelet energy analysis (store for get_status)
            if self.wavelet and len(self.data_buffer) >= 32:
                try:
                    closes = np.array([d.get('close_norm', 0) for d in list(self.data_buffer)[-64:]])
                    self.wavelet.decompose(closes)
                    self.last_wavelet_energy = self.wavelet.get_energy_distribution()
                except Exception:
                    pass

            self.total_predictions += 1
            self.last_prediction = prob_up

            return prob_up, confidence

        except Exception as e:
            logger.warning(f"LSTM prediction error: {e}")
            return 0.5, 0.0

    def _build_sequence(self, snapshots: List[Dict]) -> np.ndarray:
        """Convert list of snapshots to numpy sequence array."""
        seq = []
        for s in snapshots:
            row = [s.get(f, 0) for f in self.features]
            seq.append(row)
        return np.array(seq, dtype=np.float32)

    def _train(self):
        """Train the model on accumulated data."""
        if len(self.data_buffer) < self.sequence_length + self.prediction_horizon + 10:
            return

        try:
            sequences = []
            labels = []

            for i in range(self.sequence_length, len(self.data_buffer) - self.prediction_horizon):
                seq = self._build_sequence(self.data_buffer[i - self.sequence_length:i])
                current_close = self.data_buffer[i]['close_raw']
                future_close = self.data_buffer[i + self.prediction_horizon - 1]['close_raw']
                label = 1 if future_close > current_close else 0
                sequences.append(seq)
                labels.append(label)

            if len(sequences) < 20:
                return

            if TORCH_AVAILABLE and self.torch_model:
                self._train_torch_gpu(sequences, labels)
            elif self.numpy_model:
                self.numpy_model.train_batch(sequences, labels, epochs=5)

            self.is_trained = True
            self.samples_since_train = 0

            accuracy = self.correct_predictions / max(1, self.total_predictions)
            logger.info(f"LSTM entrenado: {len(sequences)} secuencias | "
                       f"Precision: {accuracy:.1%} ({self.total_predictions} pred) | "
                       f"Device: {self.device}")

        except Exception as e:
            logger.error(f"LSTM training error: {e}")

    def _train_torch_gpu(self, sequences: List[np.ndarray], labels: List[int]):
        """
        GPU-optimized training with DataLoader, gradient clipping,
        OneCycleLR scheduler, optional mixed precision,
        Data Augmentation, Sharpe-Aware Loss, and Curriculum Learning.
        """
        X_np = np.array(sequences)
        y_np = np.array(labels, dtype=np.float32)

        # ── Data Augmentation (x3-5 more data) ──
        if self.augmentor and len(X_np) >= 30:
            try:
                X_np, y_np = self.augmentor.augment(
                    X_np, y_np, factor=3,
                    techniques=["jitter", "scaling", "time_warp", "window_slice", "mixup"]
                )
            except Exception as e:
                logger.warning(f"Augmentation error: {e}")

        X = torch.FloatTensor(X_np).to(self.device)
        y = torch.FloatTensor(y_np).to(self.device)

        # ── Select loss function ──
        if self.sharpe_loss is not None:
            criterion = self.sharpe_loss.to(self.device)
            use_sharpe = True
        else:
            criterion = nn.BCELoss()
            use_sharpe = False

        # ── OneCycleLR scheduler ──
        epochs = self.config["epochs_per_train"]
        use_amp = self.scaler is not None

        self.torch_model.train()
        epoch_losses = []

        for epoch in range(epochs):
            # ── Curriculum Learning: progressive difficulty ──
            if self.curriculum and len(X) > 50:
                try:
                    X_epoch_np, y_epoch_np = self.curriculum.get_curriculum_data(
                        X.cpu().numpy(), y.cpu().numpy(), epoch
                    )
                    X_epoch = torch.FloatTensor(X_epoch_np).to(self.device)
                    y_epoch = torch.FloatTensor(y_epoch_np).to(self.device)
                except Exception:
                    X_epoch, y_epoch = X, y
            else:
                X_epoch, y_epoch = X, y

            dataset = TensorDataset(X_epoch, y_epoch)
            dataloader = DataLoader(
                dataset,
                batch_size=self.config["batch_size"],
                shuffle=True,
                drop_last=False,
            )

            steps_per_epoch = len(dataloader)
            if epoch == 0:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=self.config["lr"],
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    pct_start=0.3,
                    anneal_strategy='cos',
                )

            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        predictions = self.torch_model(batch_X)
                        if use_sharpe:
                            # Sharpe loss needs returns (approximate from labels)
                            returns_approx = (batch_y - 0.5) * 0.02  # Scale to return-like
                            loss = criterion(predictions, batch_y, returns_approx)
                        else:
                            loss = criterion(predictions, batch_y)

                        # Quantile loss (if quantile head exists)
                        if self.torch_model.quantile_head is not None:
                            try:
                                _, context = self.torch_model(batch_X, return_context=True)
                                q_preds = self.torch_model.quantile_head(context)
                                q_loss = self.torch_model.quantile_head.pinball_loss(
                                    q_preds, batch_y.unsqueeze(1).expand_as(q_preds)
                                )
                                loss = loss + 0.2 * q_loss  # 20% weight to quantile loss
                            except Exception:
                                pass

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.torch_model.parameters(),
                        self.config["max_grad_norm"]
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    predictions = self.torch_model(batch_X)
                    if use_sharpe:
                        returns_approx = (batch_y - 0.5) * 0.02
                        loss = criterion(predictions, batch_y, returns_approx)
                    else:
                        loss = criterion(predictions, batch_y)

                    # Quantile loss
                    if self.torch_model.quantile_head is not None:
                        try:
                            _, context = self.torch_model(batch_X, return_context=True)
                            q_preds = self.torch_model.quantile_head(context)
                            q_loss = self.torch_model.quantile_head.pinball_loss(
                                q_preds, batch_y.unsqueeze(1).expand_as(q_preds)
                            )
                            loss = loss + 0.2 * q_loss
                        except Exception:
                            pass

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.torch_model.parameters(),
                        self.config["max_grad_norm"]
                    )
                    self.optimizer.step()

                try:
                    scheduler.step()
                except Exception:
                    pass
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)

        self.last_train_loss = epoch_losses[-1] if epoch_losses else None
        self.train_losses.extend(epoch_losses[-3:])
        if len(self.train_losses) > 200:
            self.train_losses = self.train_losses[-200:]

        # Update GPU stats
        self._update_gpu_stats()

        aug_str = f" | aug=x{len(X_np)/max(len(sequences),1):.0f}" if self.augmentor else ""
        loss_type = "Sharpe+BCE" if use_sharpe else "BCE"
        logger.info(f"PyTorch GPU: {epochs} epochs | loss={self.last_train_loss:.4f} ({loss_type}) | "
                   f"samples={len(X_np)}{aug_str} | device={self.device}")

    def _update_gpu_stats(self):
        """Read GPU temperature, VRAM, utilization via pynvml."""
        if not CUDA_AVAILABLE:
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self.gpu_stats = {
                "temp_c": temp,
                "vram_used_gb": round(mem.used / 1024**3, 2),
                "vram_total_gb": round(mem.total / 1024**3, 2),
                "vram_pct": round(mem.used / mem.total * 100, 1),
                "gpu_util_pct": util.gpu,
                "mem_util_pct": util.memory,
            }
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def set_context(self, instrument: str = None, session: str = None):
        """Set instrument and session context for embeddings."""
        if instrument:
            parts = instrument.split("_")
            if len(parts) >= 3 and parts[-1] in ("M1", "M5", "M15", "M30", "H1", "H4", "D", "W"):
                instrument = "_".join(parts[:-1])
            self.current_instrument_id = INSTRUMENT_IDS.get(instrument, 0)
        if session:
            self.current_session_id = SESSION_IDS.get(session, 5)

    def get_attention_summary(self) -> Dict:
        """Return attention summary with per-head detail for dashboard."""
        if self.last_attention_weights is None:
            return {"available": False}

        w = self.last_attention_weights
        seq_len = len(w)
        top_3_idx = np.argsort(w)[-3:][::-1]

        result = {
            "available": True,
            "seq_len": seq_len,
            "top_bars": [{"bar_ago": seq_len - 1 - int(i), "weight": round(float(w[i]), 4)} for i in top_3_idx],
            "concentration": round(float(np.max(w)), 4),
            "entropy": round(float(-np.sum(w * np.log(w + 1e-10))), 3),
            "weights": [round(float(x), 4) for x in w],
            "num_heads": self.config.get("num_heads", 1),
        }

        # Per-head detail
        if self.last_head_weights is not None:
            heads = []
            for h in range(self.last_head_weights.shape[0]):
                hw = self.last_head_weights[h]
                top_idx = np.argsort(hw)[-2:][::-1]
                heads.append({
                    "head": h,
                    "top_bars": [{"bar_ago": seq_len - 1 - int(i), "weight": round(float(hw[i]), 4)} for i in top_idx],
                    "max_weight": round(float(np.max(hw)), 4),
                })
            result["heads"] = heads

        return result

    def get_gpu_status(self) -> Dict:
        """Return GPU stats for dashboard."""
        self._update_gpu_stats()
        return {
            "cuda_available": CUDA_AVAILABLE,
            "device": self.device,
            "gpu_name": GPU_NAME if CUDA_AVAILABLE else "N/A",
            "gpu_vram_gb": GPU_VRAM if CUDA_AVAILABLE else 0,
            **self.gpu_stats,
            "last_train_loss": self.last_train_loss,
            "train_loss_history": self.train_losses[-50:],
            "model_params": sum(p.numel() for p in self.torch_model.parameters()) if self.torch_model else 0,
        }

    def get_status(self) -> Dict:
        """Get predictor status for dashboard."""
        accuracy = self.correct_predictions / max(1, self.total_predictions)
        attn = self.get_attention_summary()

        backend = "NumpyGRU+Attn"
        if TORCH_AVAILABLE:
            backend = f"PyTorch+MHA({self.config['num_heads']}h)"
            if CUDA_AVAILABLE:
                backend = f"CUDA+MHA({self.config['num_heads']}h)"

        result = {
            "is_trained": self.is_trained,
            "backend": backend,
            "device": self.device,
            "buffer_size": len(self.data_buffer),
            "total_predictions": self.total_predictions,
            "accuracy": accuracy,
            "last_prediction": self.last_prediction,
            "hidden_size": self.hidden_size,
            "sequence_length": self.sequence_length,
            "num_heads": self.config.get("num_heads", 1),
            "num_layers": self.config.get("num_layers", 1),
            "attention": attn,
            "last_train_loss": self.last_train_loss,
            "gpu": self.gpu_stats if CUDA_AVAILABLE else {},
        }

        # v4 Advanced Learning status
        if self.mc_dropout:
            result["mc_dropout"] = self.mc_dropout.get_status()
        if self.last_uncertainty:
            result["uncertainty"] = self.last_uncertainty
        if self.last_quantiles:
            result["quantiles"] = self.last_quantiles
        result["advanced_learning"] = {
            "sharpe_loss": self.sharpe_loss is not None,
            "mc_dropout": self.mc_dropout is not None,
            "data_augmentation": self.augmentor is not None,
            "curriculum_learning": self.curriculum is not None,
            "quantile_regression": (self.torch_model and
                                    hasattr(self.torch_model, 'quantile_head') and
                                    self.torch_model.quantile_head is not None) if TORCH_AVAILABLE else False,
            "positional_encoding": self.pos_encoding is not None,
            # v8 strategies
            "wavelet_decomposition": self.wavelet is not None,
            "wasserstein_drift": self.wasserstein_drift is not None,
            "contrastive_learning": self.contrastive_learner is not None,
            "market_vae": self.market_vae is not None,
        }

        # v8 status details
        if self.last_wavelet_energy:
            result["wavelet_energy"] = self.last_wavelet_energy
        if self.last_drift_status:
            result["wasserstein_drift"] = self.last_drift_status
        if self.last_vae_anomaly is not None:
            result["vae_anomaly"] = self.last_vae_anomaly

        return result

    def save_model(self):
        """Save model weights to disk."""
        if self.torch_model and TORCH_AVAILABLE:
            try:
                torch.save({
                    'model_state': self.torch_model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'config': self.config,
                    'is_trained': self.is_trained,
                    'total_predictions': self.total_predictions,
                    'correct_predictions': self.correct_predictions,
                }, self.model_path)
                logger.info(f"Modelo guardado: {self.model_path}")
            except Exception as e:
                logger.error(f"Error guardando modelo: {e}")

    def load_model(self):
        """Load model weights from disk."""
        if not os.path.exists(self.model_path):
            return False
        if self.torch_model and TORCH_AVAILABLE:
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.torch_model.load_state_dict(checkpoint['model_state'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                self.is_trained = checkpoint.get('is_trained', False)
                self.total_predictions = checkpoint.get('total_predictions', 0)
                self.correct_predictions = checkpoint.get('correct_predictions', 0)
                logger.info(f"Modelo cargado: {self.model_path}")
                return True
            except Exception as e:
                logger.warning(f"Error cargando modelo: {e}")
        return False
