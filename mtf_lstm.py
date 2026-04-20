"""
ML SuperTrend v51 - Multi-Timeframe LSTM
==========================================
Procesa múltiples timeframes (M5, M15, H1, H4) en paralelo
con fusión cross-temporal mediante attention.

Arquitectura:
  TF Encoders (parallel):
    M5  → LSTM_M5(64h)  → h_m5
    M15 → LSTM_M15(64h) → h_m15
    H1  → LSTM_H1(64h)  → h_h1
    H4  → LSTM_H4(64h)  → h_h4

  Cross-Temporal Fusion:
    [h_m5, h_m15, h_h1, h_h4] → MultiHead Attention → fused

  Predictor:
    fused → Dense(64) → ReLU → Dense(1) → Sigmoid → P(UP)

Papers:
  - "Multi-Scale Temporal Fusion" (Lim et al., 2021)
  - "Hierarchical Multi-Scale Networks" (Chen et al., 2022)

Uso:
    from mtf_lstm import MultiTimeframeLSTM
    mtf = MultiTimeframeLSTM()
    mtf.update_timeframe("M5", candles_m5, indicators_m5)
    mtf.update_timeframe("H1", candles_h1, indicators_h1)
    prob, confidence = mtf.predict()
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    pass


# =====================================================================
# TORCH MULTI-TIMEFRAME MODEL
# =====================================================================

if TORCH_AVAILABLE:

    class TimeframeEncoder(nn.Module):
        """LSTM encoder para un timeframe individual."""

        def __init__(self, input_dim: int = 6, hidden_dim: int = 64,
                     num_layers: int = 1, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args: x (batch, seq_len, input_dim)
            Returns: hidden (batch, hidden_dim)
            """
            output, (h_n, _) = self.lstm(x)
            # Use last hidden state
            h = h_n[-1]  # (batch, hidden_dim)
            return self.norm(h)


    class CrossTemporalAttention(nn.Module):
        """
        Multi-Head Attention entre timeframes.
        Permite que M5 atienda a H1, H4 atienda a M15, etc.
        """

        def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
            super().__init__()
            self.attention = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.last_weights = None

        def forward(self, tf_embeddings: torch.Tensor) -> torch.Tensor:
            """
            Args: tf_embeddings (batch, n_timeframes, d_model)
            Returns: fused (batch, d_model)
            """
            attn_out, weights = self.attention(
                tf_embeddings, tf_embeddings, tf_embeddings
            )
            self.last_weights = weights.detach()

            # Residual + norm
            fused = self.norm(tf_embeddings + self.dropout(attn_out))

            # Average pool across timeframes
            return fused.mean(dim=1)  # (batch, d_model)


    class MultiTimeframeModel(nn.Module):
        """
        Full multi-timeframe model.
        N timeframe encoders → cross-temporal attention → prediction head.
        """

        def __init__(
            self,
            n_timeframes: int = 4,
            input_dim: int = 6,
            hidden_dim: int = 64,
            n_heads: int = 4,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.n_timeframes = n_timeframes
            self.hidden_dim = hidden_dim

            # One encoder per timeframe
            self.encoders = nn.ModuleList([
                TimeframeEncoder(input_dim, hidden_dim, num_layers=1, dropout=dropout)
                for _ in range(n_timeframes)
            ])

            # Timeframe embedding (learnable positional encoding for TF hierarchy)
            self.tf_embedding = nn.Embedding(n_timeframes, hidden_dim)

            # Cross-temporal fusion
            self.cross_attention = CrossTemporalAttention(hidden_dim, n_heads, dropout)

            # Prediction head
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

            # Timeframe importance (learnable weights)
            self.tf_importance = nn.Parameter(torch.ones(n_timeframes) / n_timeframes)

        def forward(self, tf_sequences: List[torch.Tensor]) -> torch.Tensor:
            """
            Args:
                tf_sequences: list of (batch, seq_len_i, input_dim) for each TF

            Returns:
                prob_up: (batch, 1)
            """
            batch_size = tf_sequences[0].shape[0]

            # Encode each timeframe
            tf_hiddens = []
            for i, (encoder, seq) in enumerate(zip(self.encoders, tf_sequences)):
                h = encoder(seq)  # (batch, hidden_dim)
                # Add timeframe positional embedding
                tf_pos = self.tf_embedding(
                    torch.tensor([i], device=seq.device)
                ).expand(batch_size, -1)
                h = h + tf_pos
                tf_hiddens.append(h)

            # Stack: (batch, n_timeframes, hidden_dim)
            tf_stack = torch.stack(tf_hiddens, dim=1)

            # Apply learnable importance weights
            importance = torch.softmax(self.tf_importance, dim=0)
            tf_stack = tf_stack * importance.unsqueeze(0).unsqueeze(-1)

            # Cross-temporal attention
            fused = self.cross_attention(tf_stack)  # (batch, hidden_dim)

            # Predict
            return self.predictor(fused)

        def get_tf_importance(self) -> Dict[str, float]:
            """Get learned timeframe importance weights."""
            weights = torch.softmax(self.tf_importance, dim=0).detach().cpu().numpy()
            tf_names = ["M5", "M15", "H1", "H4"]
            return {
                tf_names[i] if i < len(tf_names) else f"TF_{i}": round(float(w), 4)
                for i, w in enumerate(weights)
            }

else:
    # Placeholders
    class TimeframeEncoder:
        def __init__(self, *args, **kwargs): pass

    class CrossTemporalAttention:
        def __init__(self, *args, **kwargs): pass

    class MultiTimeframeModel:
        def __init__(self, *args, **kwargs): pass


# =====================================================================
# HIGH-LEVEL WRAPPER
# =====================================================================

class MultiTimeframeLSTM:
    """
    Wrapper de alto nivel para Multi-Timeframe LSTM.
    Gestiona buffers de datos para cada TF y coordina predicciones.
    """

    TIMEFRAMES = ["M5", "M15", "H1", "H4"]
    TF_SEQ_LENGTHS = {"M5": 30, "M15": 20, "H1": 15, "H4": 10}
    INPUT_FEATURES = 6  # close_norm, rsi, adx, macd, atr, vol

    def __init__(
        self,
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.2,
        min_samples_per_tf: int = 50,
    ):
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.min_samples = min_samples_per_tf

        # Data buffers per timeframe
        self.buffers: Dict[str, deque] = {
            tf: deque(maxlen=500) for tf in self.TIMEFRAMES
        }

        # Model
        self.model = None
        self.device = "cpu"
        self.is_trained = False
        self.last_prediction = None
        self.last_tf_importance = None
        self.last_cross_attention = None

        if TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = MultiTimeframeModel(
                n_timeframes=len(self.TIMEFRAMES),
                input_dim=self.INPUT_FEATURES,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
            ).to(self.device)
            logger.info(f"MultiTimeframeLSTM initialized ({self.device})")

    def update_timeframe(self, timeframe: str, candles: List[Dict], indicators: Dict):
        """
        Alimentar datos de un timeframe específico.

        Args:
            timeframe: "M5", "M15", "H1", "H4"
            candles: Lista de candles
            indicators: Dict con rsi, adx, macd, atr, vol_ratio
        """
        if timeframe not in self.buffers:
            logger.warning(f"Unknown timeframe: {timeframe}")
            return

        if not candles or not indicators:
            return

        try:
            closes = indicators.get('close', np.array([0]))
            rsi = indicators.get('rsi', np.array([50]))
            adx = indicators.get('adx', np.array([20]))
            macd = indicators.get('macd_histogram', np.array([0]))
            atr = indicators.get('atr', np.array([0]))
            vol = indicators.get('vol_ratio', np.array([1]))

            close = float(closes[-1])
            close_mean = float(np.mean(closes[-50:])) if len(closes) >= 50 else close
            close_std = float(np.std(closes[-50:])) if len(closes) >= 50 else max(close * 0.01, 1e-8)

            snapshot = {
                'close_norm': (close - close_mean) / close_std if close_std > 0 else 0,
                'rsi_norm': float(rsi[-1]) / 100.0,
                'adx_norm': min(float(adx[-1]) / 50.0, 1.0),
                'macd_norm': np.clip(float(macd[-1]) / (close * 0.001 + 1e-8), -1, 1),
                'atr_norm': min(float(atr[-1]) / (close * 0.01 + 1e-8), 1.0),
                'vol_norm': min(float(vol[-1]), 3.0) / 3.0,
            }

            self.buffers[timeframe].append(snapshot)

        except Exception as e:
            logger.debug(f"MTF update error ({timeframe}): {e}")

    def predict(self) -> Tuple[float, float]:
        """
        Predecir usando todos los timeframes disponibles.

        Returns:
            (prob_up, confidence)
        """
        if not self.model or not self.is_trained:
            return 0.5, 0.0

        # Check all TFs have enough data
        tf_sequences = []
        available_tfs = []

        for tf in self.TIMEFRAMES:
            seq_len = self.TF_SEQ_LENGTHS.get(tf, 15)
            if len(self.buffers[tf]) >= seq_len:
                data = list(self.buffers[tf])[-seq_len:]
                seq = np.array([[
                    d['close_norm'], d['rsi_norm'], d['adx_norm'],
                    d['macd_norm'], d['atr_norm'], d['vol_norm']
                ] for d in data])
                tf_sequences.append(seq)
                available_tfs.append(tf)
            else:
                # Pad with zeros if not enough data
                seq_len_needed = self.TF_SEQ_LENGTHS.get(tf, 15)
                tf_sequences.append(np.zeros((seq_len_needed, self.INPUT_FEATURES)))
                available_tfs.append(f"{tf}(padded)")

        if not tf_sequences:
            return 0.5, 0.0

        try:
            self.model.eval()
            with torch.no_grad():
                tensors = [
                    torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                    for seq in tf_sequences
                ]
                prob = float(self.model(tensors).cpu().item())

            confidence = abs(prob - 0.5) * 2
            # Reduce confidence if some TFs are padded
            n_padded = sum(1 for tf in available_tfs if "padded" in tf)
            if n_padded > 0:
                confidence *= (1 - n_padded / len(self.TIMEFRAMES) * 0.5)

            self.last_prediction = prob
            self.last_tf_importance = self.model.get_tf_importance()

            return prob, confidence

        except Exception as e:
            logger.warning(f"MTF prediction error: {e}")
            return 0.5, 0.0

    def train(self, tf_data: Dict[str, Tuple[np.ndarray, np.ndarray]], epochs: int = 10):
        """
        Entrenar el modelo multi-timeframe.

        Args:
            tf_data: {timeframe: (X_sequences, y_labels)} para cada TF
            epochs: Número de épocas
        """
        if not self.model or not TORCH_AVAILABLE:
            return

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        # Get minimum number of samples across all TFs
        min_n = min(len(v[1]) for v in tf_data.values()) if tf_data else 0
        if min_n < 10:
            logger.warning("Not enough multi-TF data for training")
            return

        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0

            for batch_start in range(0, min_n - 32, 32):
                batch_end = min(batch_start + 32, min_n)

                tf_tensors = []
                for tf in self.TIMEFRAMES:
                    if tf in tf_data:
                        X_batch = tf_data[tf][0][batch_start:batch_end]
                        tf_tensors.append(torch.FloatTensor(X_batch).to(self.device))
                    else:
                        seq_len = self.TF_SEQ_LENGTHS.get(tf, 15)
                        tf_tensors.append(
                            torch.zeros(batch_end - batch_start, seq_len, self.INPUT_FEATURES).to(self.device)
                        )

                # Use labels from first available TF
                first_tf = list(tf_data.keys())[0]
                y = torch.FloatTensor(
                    tf_data[first_tf][1][batch_start:batch_end]
                ).unsqueeze(1).to(self.device)

                optimizer.zero_grad()
                pred = self.model(tf_tensors)
                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            if epoch % 5 == 0:
                logger.info(f"MTF Train epoch {epoch}: loss={avg_loss:.4f}")

        self.is_trained = True
        logger.info(f"MTF training complete: {epochs} epochs")

    def get_status(self) -> Dict:
        """Estado para dashboard."""
        buffer_sizes = {tf: len(buf) for tf, buf in self.buffers.items()}
        return {
            "available": self.model is not None,
            "is_trained": self.is_trained,
            "device": self.device,
            "timeframes": self.TIMEFRAMES,
            "buffer_sizes": buffer_sizes,
            "hidden_dim": self.hidden_dim,
            "n_heads": self.n_heads,
            "last_prediction": self.last_prediction,
            "tf_importance": self.last_tf_importance,
        }
