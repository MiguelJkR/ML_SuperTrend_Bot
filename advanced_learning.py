"""
ML SuperTrend v51 - Advanced Learning Strategies
==================================================
Módulo consolidado de estrategias científicas de aprendizaje:

1. Sharpe-Aware Loss: Optimizar directamente el ratio de Sharpe
2. Quantile Regression: Predecir distribución completa de retornos
3. MC Dropout (Bayesian Uncertainty): Incertidumbre en predicciones
4. TD-λ Learning: Retornos descontados con horizonte variable
5. Fisher Information: Detección rápida de cambio de régimen
6. Curriculum Learning: Entrenamiento progresivo fácil→difícil
7. Financial Positional Encoding: Contexto temporal financiero
8. Online Learning EXP3: Pesos adaptativos en tiempo real

Papers base:
  - Moody & Saffell (1998), Zhang et al. (2020) — Sharpe Loss
  - Koenker & Bassett (1978) — Quantile Regression
  - Gal & Ghahramani (2016) — MC Dropout
  - Sutton (1988) — TD-λ
  - Bengio et al. (2009) — Curriculum Learning
  - Auer et al. (2002) — EXP3

Uso:
    from advanced_learning import (
        SharpeLoss, QuantileHead, MCDropoutPredictor,
        TDLambdaEvaluator, FisherChangeDetector,
        CurriculumScheduler, FinancialPositionalEncoding,
        EXP3OnlineLearner
    )
"""

import logging
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from collections import deque

logger = logging.getLogger(__name__)

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    pass


# =====================================================================
# 1. SHARPE-AWARE LOSS FUNCTION
# L = -(mean(R) / std(R)) × √252
# Optimiza directamente el ratio de Sharpe diferenciable
# =====================================================================

if TORCH_AVAILABLE:
    class SharpeLoss(nn.Module):
        """
        Differentiable Sharpe Ratio Loss.
        Maximiza el ratio de Sharpe de los retornos predichos.

        L = -(mean(R) / (std(R) + ε)) × √annualization_factor

        Combina con BCE para mantener señal de clasificación:
          total_loss = α × BCE + (1-α) × (-SharpeRatio)
        """

        def __init__(self, annualization: float = 252.0, alpha: float = 0.7, epsilon: float = 1e-6):
            super().__init__()
            self.annualization = annualization
            self.alpha = alpha  # Peso BCE vs Sharpe
            self.epsilon = epsilon
            self.bce = nn.BCELoss()

        def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                    returns: torch.Tensor = None) -> torch.Tensor:
            """
            Args:
                predictions: (batch,) probabilidades predichas
                targets: (batch,) labels binarios (0/1)
                returns: (batch,) retornos reales (opcional, para Sharpe puro)
            """
            # BCE component (siempre)
            bce_loss = self.bce(predictions, targets)

            if returns is not None and len(returns) > 2:
                # Sharpe component usando retornos reales ponderados por predicción
                # R_weighted = prediction × return (signal-weighted returns)
                weighted_returns = predictions * returns
                mean_r = weighted_returns.mean()
                std_r = weighted_returns.std() + self.epsilon
                sharpe = mean_r / std_r * (self.annualization ** 0.5)
                sharpe_loss = -sharpe  # Negativo porque queremos MAXIMIZAR

                total = self.alpha * bce_loss + (1 - self.alpha) * sharpe_loss
            else:
                total = bce_loss

            return total


# =====================================================================
# 2. QUANTILE REGRESSION HEAD
# Predice cuantiles 10%, 25%, 50%, 75%, 90%
# Pinball Loss: L_q(y, ŷ) = q×max(y-ŷ,0) + (1-q)×max(ŷ-y,0)
# =====================================================================

if TORCH_AVAILABLE:
    class QuantileHead(nn.Module):
        """
        Cabeza de regresión cuantílica.
        Predice múltiples cuantiles de la distribución de retornos.

        Cuantiles: [0.10, 0.25, 0.50, 0.75, 0.90]
        El cuantil 10% sirve como worst-case para risk management.
        El cuantil 50% es la predicción central (mediana).
        El spread 90%-10% indica incertidumbre.
        """

        def __init__(self, input_size: int = 128, quantiles: List[float] = None):
            super().__init__()
            self.quantiles = quantiles or [0.10, 0.25, 0.50, 0.75, 0.90]
            self.n_quantiles = len(self.quantiles)

            self.head = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, self.n_quantiles),  # Un output por cuantil
            )
            self.last_quantiles = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            x: (batch, input_size) — features del modelo base
            Returns: (batch, n_quantiles) — predicciones por cuantil
            """
            quantile_preds = self.head(x)
            self.last_quantiles = quantile_preds.detach()
            return quantile_preds

        def pinball_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """
            Pinball (Quantile) Loss.
            L_q(y, ŷ) = q × max(y - ŷ, 0) + (1 - q) × max(ŷ - y, 0)
            """
            total_loss = torch.tensor(0.0, device=predictions.device)

            for i, q in enumerate(self.quantiles):
                pred_q = predictions[:, i]
                error = targets - pred_q
                loss_q = torch.mean(torch.max(q * error, (q - 1) * error))
                total_loss += loss_q

            return total_loss / self.n_quantiles

        def get_risk_metrics(self) -> Optional[Dict]:
            """Extraer métricas de riesgo de la última predicción."""
            if self.last_quantiles is None:
                return None

            q = self.last_quantiles[-1].cpu().numpy()  # Última muestra
            return {
                "q10_worst_case": float(q[0]),
                "q25": float(q[1]),
                "q50_median": float(q[2]),
                "q75": float(q[3]),
                "q90_best_case": float(q[4]),
                "iqr": float(q[3] - q[1]),          # Interquartile range
                "uncertainty": float(q[4] - q[0]),   # 80% prediction interval
                "skew_indicator": float((q[4] - q[2]) - (q[2] - q[0])),  # Asimetría
            }


# =====================================================================
# 3. MC DROPOUT - BAYESIAN UNCERTAINTY
# Mantener dropout activo en inferencia, N forward passes
# uncertainty = std(f(x, mask_1), ..., f(x, mask_N))
# Gal & Ghahramani (2016)
# =====================================================================

class MCDropoutPredictor:
    """
    Monte Carlo Dropout para estimación de incertidumbre.
    Deja dropout ACTIVO durante inferencia y hace N forward passes.

    Si la incertidumbre es alta → señal de NO operar.
    """

    def __init__(
        self,
        n_forward_passes: int = 20,
        uncertainty_threshold: float = 0.15,  # Si std > esto → no operar
    ):
        self.n_forward_passes = n_forward_passes
        self.uncertainty_threshold = uncertainty_threshold
        self.last_mean = 0.5
        self.last_std = 0.0
        self.last_predictions = []

    def predict_with_uncertainty(self, model, x_tensor) -> Dict:
        """
        Hacer N forward passes con dropout activo.

        Args:
            model: nn.Module con capas de Dropout
            x_tensor: input tensor (1, seq_len, features)

        Returns:
            {
                "mean": float,         # Predicción promedio
                "std": float,          # Incertidumbre
                "confidence": float,   # 1 - uncertainty_normalized
                "should_trade": bool,  # Si la incertidumbre es aceptable
                "predictions": list,   # Todas las predicciones
                "q5": float,           # Percentil 5 (worst case)
                "q95": float,          # Percentil 95 (best case)
            }
        """
        if not TORCH_AVAILABLE:
            return {
                "mean": 0.5, "std": 0.0, "confidence": 0.0,
                "should_trade": False, "predictions": [],
                "q5": 0.5, "q95": 0.5,
            }

        # CLAVE: poner en modo train para mantener dropout activo
        model.train()

        predictions = []
        with torch.no_grad():
            for _ in range(self.n_forward_passes):
                # Cada forward pass usa una máscara de dropout diferente
                pred = model(x_tensor)
                predictions.append(float(pred.cpu().item()))

        # Restaurar modo eval (para operación normal fuera de MC)
        model.eval()

        predictions = np.array(predictions)
        mean_pred = float(np.mean(predictions))
        std_pred = float(np.std(predictions))

        # Normalizar incertidumbre (0 a 1)
        uncertainty_norm = min(std_pred / 0.3, 1.0)
        confidence = 1.0 - uncertainty_norm
        should_trade = std_pred <= self.uncertainty_threshold

        self.last_mean = mean_pred
        self.last_std = std_pred
        self.last_predictions = predictions.tolist()

        return {
            "mean": round(mean_pred, 4),
            "std": round(std_pred, 4),
            "confidence": round(confidence, 4),
            "should_trade": should_trade,
            "predictions": [round(p, 4) for p in predictions[:5]],  # Solo primeras 5 para dashboard
            "q5": round(float(np.percentile(predictions, 5)), 4),
            "q95": round(float(np.percentile(predictions, 95)), 4),
            "n_passes": self.n_forward_passes,
        }

    def get_status(self) -> Dict:
        return {
            "n_passes": self.n_forward_passes,
            "threshold": self.uncertainty_threshold,
            "last_mean": round(self.last_mean, 4),
            "last_std": round(self.last_std, 4),
            "last_should_trade": self.last_std <= self.uncertainty_threshold,
        }


# =====================================================================
# 4. TD-λ LEARNING
# V(s_t) = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γⁿ·r_{t+n}
# Con λ para ponderar entre TD(0) y Monte Carlo
# Sutton (1988)
# =====================================================================

class TDLambdaEvaluator:
    """
    Temporal Difference Learning con λ-returns.
    Evalúa trades con retornos descontados en vez de binario win/loss.

    Ventajas:
      - Captura calidad de la señal, no solo el resultado final
      - Pondera trades recientes más que antiguos
      - Da crédito parcial a señales que fueron correctas pero timing imperfecto
    """

    def __init__(
        self,
        gamma: float = 0.97,      # Factor de descuento (cuánto vale el futuro)
        lambda_: float = 0.95,    # λ para mezclar TD(0) y MC
        normalize: bool = True,   # Normalizar retornos
    ):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.normalize = normalize
        self.value_estimates: List[float] = []
        self.reward_history: deque = deque(maxlen=1000)

    def compute_td_lambda_returns(self, rewards: List[float]) -> np.ndarray:
        """
        Calcular λ-returns para una secuencia de recompensas.

        G_t^λ = (1-λ) Σ_{n=1}^{T-t-1} λ^{n-1} G_t^{(n)} + λ^{T-t-1} G_t^{(T-t)}

        Simplificado: usamos la formulación recursiva hacia atrás.
        """
        T = len(rewards)
        if T == 0:
            return np.array([])

        rewards = np.array(rewards, dtype=np.float64)

        if self.normalize and np.std(rewards) > 0:
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

        # TD(λ) returns calculados hacia atrás
        td_returns = np.zeros(T, dtype=np.float64)
        td_returns[-1] = rewards[-1]

        for t in range(T - 2, -1, -1):
            # G_t = r_t + γ × [(1-λ) × V(s_{t+1}) + λ × G_{t+1}]
            td_returns[t] = rewards[t] + self.gamma * (
                (1 - self.lambda_) * rewards[t + 1] +
                self.lambda_ * td_returns[t + 1]
            )

        return td_returns

    def evaluate_trade_sequence(self, trade_results: List[Dict]) -> List[Dict]:
        """
        Evaluar una secuencia de trades con TD-λ en vez de win/loss binario.

        Args:
            trade_results: lista de {pnl, duration, instrument, ...}

        Returns:
            Lista con td_value agregado a cada trade
        """
        if not trade_results:
            return []

        # Extraer recompensas (PnL normalizado)
        rewards = [t.get("pnl", 0) for t in trade_results]
        td_returns = self.compute_td_lambda_returns(rewards)

        # Agregar TD value a cada trade
        evaluated = []
        for i, trade in enumerate(trade_results):
            t = trade.copy()
            t["td_value"] = round(float(td_returns[i]), 6)
            t["td_rank"] = "excellent" if td_returns[i] > 0.5 else \
                           "good" if td_returns[i] > 0 else \
                           "poor" if td_returns[i] > -0.5 else "bad"
            evaluated.append(t)

        self.value_estimates = td_returns.tolist()
        return evaluated

    def get_cumulative_td_value(self) -> float:
        """Valor TD acumulado (indicador de calidad general del modelo)."""
        if not self.value_estimates:
            return 0.0
        return float(np.mean(self.value_estimates[-50:]))

    def get_status(self) -> Dict:
        return {
            "gamma": self.gamma,
            "lambda": self.lambda_,
            "n_evaluated": len(self.value_estimates),
            "cumulative_td": round(self.get_cumulative_td_value(), 4),
            "recent_td_values": [round(v, 4) for v in self.value_estimates[-10:]],
        }


# =====================================================================
# 5. FISHER INFORMATION — DETECCIÓN DE CAMBIO DE RÉGIMEN
# F(θ) = E[(∂log p(x|θ)/∂θ)²]
# Cuando F sube bruscamente → mercado cambiando
# =====================================================================

class FisherChangeDetector:
    """
    Detección de cambio de régimen usando Fisher Information.
    Mide cuánto cambia la distribución del mercado.

    Cuando Fisher Information sube → el mercado está cambiando
    → Reducir posiciones automáticamente.

    Más rápido que indicadores técnicos tradicionales.
    """

    def __init__(
        self,
        window: int = 30,           # Ventana para calcular distribución
        alert_threshold: float = 2.0,  # Multiplicador sobre media para alertar
        critical_threshold: float = 3.0,  # Multiplicador para acción inmediata
    ):
        self.window = window
        self.alert_threshold = alert_threshold
        self.critical_threshold = critical_threshold

        self.returns_buffer: deque = deque(maxlen=window * 3)
        self.fisher_history: deque = deque(maxlen=200)
        self.current_fisher = 0.0
        self.fisher_mean = 0.0
        self.alert_level = "NORMAL"  # NORMAL, ALERT, CRITICAL

    def update(self, returns: float):
        """Alimentar con nuevo retorno y recalcular Fisher Information."""
        self.returns_buffer.append(returns)

        if len(self.returns_buffer) < self.window:
            return

        # Calcular Fisher Information
        recent = np.array(list(self.returns_buffer))

        # Dividir en dos mitades para detectar cambio
        half = len(recent) // 2
        first_half = recent[:half]
        second_half = recent[half:]

        # Fisher Information ≈ cambio en parámetros de la distribución
        # Para gaussiana: F = 1/σ² + (μ₁-μ₂)²/σ²
        mu1, sig1 = np.mean(first_half), np.std(first_half) + 1e-8
        mu2, sig2 = np.mean(second_half), np.std(second_half) + 1e-8

        # Fisher divergence (simplificada)
        fisher = ((mu1 - mu2) ** 2) / (sig1 * sig2) + \
                 0.5 * ((sig1 / sig2) ** 2 + (sig2 / sig1) ** 2 - 2)

        self.current_fisher = float(fisher)
        self.fisher_history.append(self.current_fisher)

        # Actualizar media móvil de Fisher
        if len(self.fisher_history) > 10:
            self.fisher_mean = float(np.mean(list(self.fisher_history)))

        # Determinar nivel de alerta
        if self.fisher_mean > 0:
            ratio = self.current_fisher / (self.fisher_mean + 1e-8)
            if ratio >= self.critical_threshold:
                self.alert_level = "CRITICAL"
            elif ratio >= self.alert_threshold:
                self.alert_level = "ALERT"
            else:
                self.alert_level = "NORMAL"

    def get_risk_multiplier(self) -> float:
        """
        Multiplicador de riesgo basado en Fisher Information.
        - NORMAL: 1.0
        - ALERT: 0.6 (reducir 40%)
        - CRITICAL: 0.3 (reducir 70%)
        """
        multipliers = {"NORMAL": 1.0, "ALERT": 0.6, "CRITICAL": 0.3}
        return multipliers.get(self.alert_level, 1.0)

    def is_regime_changing(self) -> bool:
        """¿El mercado está cambiando de régimen?"""
        return self.alert_level in ("ALERT", "CRITICAL")

    def get_status(self) -> Dict:
        return {
            "current_fisher": round(self.current_fisher, 6),
            "fisher_mean": round(self.fisher_mean, 6),
            "alert_level": self.alert_level,
            "risk_multiplier": self.get_risk_multiplier(),
            "is_changing": self.is_regime_changing(),
            "buffer_size": len(self.returns_buffer),
            "fisher_history": [round(f, 4) for f in list(self.fisher_history)[-20:]],
        }


# =====================================================================
# 6. CURRICULUM LEARNING
# difficulty(sample) = |retorno| / volatilidad
# Entrenar primero con datos fáciles, progresivamente agregar difíciles
# Bengio et al. (2009)
# =====================================================================

class CurriculumScheduler:
    """
    Curriculum Learning para entrenamiento progresivo.
    Ordena datos por dificultad y entrena gradualmente.

    Dificultad de un sample:
      - Fácil: tendencia clara con baja volatilidad (|return|/vol alto)
      - Difícil: consolidación con alta volatilidad (|return|/vol bajo)

    Convergencia 30-40% más rápida según Bengio et al. (2009).
    """

    def __init__(
        self,
        initial_pct: float = 0.3,    # Empezar con 30% más fácil
        growth_rate: float = 0.1,     # Agregar 10% por época
        min_difficulty_pct: float = 1.0,  # Usar 100% al final
    ):
        self.initial_pct = initial_pct
        self.growth_rate = growth_rate
        self.min_difficulty_pct = min_difficulty_pct
        self.current_epoch = 0

    def get_curriculum_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epoch: int,
        returns: np.ndarray = None,
        volatilities: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Seleccionar subset de datos según dificultad y época actual.

        Args:
            X: (n_samples, seq_len, features)
            y: (n_samples,) labels
            epoch: época actual de entrenamiento
            returns: (n_samples,) retornos para calcular dificultad
            volatilities: (n_samples,) volatilidades

        Returns:
            X_curriculum, y_curriculum (subsets ordenados por dificultad)
        """
        n = len(X)

        if returns is None or volatilities is None:
            # Heurística: usar varianza de la secuencia como proxy de dificultad
            difficulties = np.array([np.std(X[i, :, 0]) for i in range(n)])
        else:
            # Dificultad = |retorno| / volatilidad (inverso porque queremos fácil primero)
            # Alto |ret|/vol = fácil (tendencia clara), Bajo = difícil (consolidación)
            signal_to_noise = np.abs(returns) / (volatilities + 1e-8)
            # Invertir: queremos que "fácil" tenga dificultad BAJA
            difficulties = 1.0 / (signal_to_noise + 1e-8)

        # Porcentaje de datos a usar en esta época
        pct = min(self.initial_pct + epoch * self.growth_rate, self.min_difficulty_pct)
        n_select = max(int(n * pct), 10)

        # Ordenar por dificultad (menor primero = más fácil)
        sorted_indices = np.argsort(difficulties)
        selected = sorted_indices[:n_select]

        # Shuffle para evitar sesgo de orden
        np.random.shuffle(selected)

        self.current_epoch = epoch
        return X[selected], y[selected]

    def get_difficulty_stats(self, X: np.ndarray) -> Dict:
        """Estadísticas de dificultad del dataset."""
        difficulties = np.array([np.std(X[i, :, 0]) for i in range(len(X))])
        return {
            "n_samples": len(X),
            "mean_difficulty": round(float(np.mean(difficulties)), 4),
            "std_difficulty": round(float(np.std(difficulties)), 4),
            "easy_pct": round(float(np.mean(difficulties < np.median(difficulties))) * 100, 1),
            "current_epoch": self.current_epoch,
            "current_pct": min(self.initial_pct + self.current_epoch * self.growth_rate, 1.0),
        }


# =====================================================================
# 7. FINANCIAL POSITIONAL ENCODING
# PE_fin(pos) = [día_semana, hora_sesión, días_hasta_NFP, días_hasta_FOMC]
# Combinado con sinusoidal estándar
# =====================================================================

class FinancialPositionalEncoding:
    """
    Encoding posicional financiero que captura patrones temporales del mercado.

    Componentes:
      1. Sinusoidal estándar: PE(pos, 2i) = sin(pos/10000^(2i/d))
      2. Día de la semana (one-hot 5 dims)
      3. Hora de sesión (normalizada 0-1)
      4. Sesión de trading (ASIAN/LONDON/NY, one-hot 3 dims)
      5. Cercanía a eventos macro (NFP, FOMC, GDP — decay exponencial)

    Total: d_model + 5 + 1 + 3 + 3 = d_model + 12 features adicionales
    """

    # Fechas aproximadas de eventos macro mensuales (día del mes típico)
    NFP_DAY = 5        # First Friday (aprox día 5)
    FOMC_DAY = 15      # Mid-month (aprox día 15)
    GDP_DAY = 25        # Late month (aprox día 25)

    # Sesiones de trading (horas UTC)
    SESSIONS = {
        "ASIAN": (0, 8),
        "LONDON": (7, 16),
        "NEW_YORK": (13, 22),
    }

    def __init__(self, d_model: int = 6, max_len: int = 100):
        self.d_model = d_model
        self.max_len = max_len

    def encode(self, timestamps: List[datetime] = None, seq_len: int = 30) -> np.ndarray:
        """
        Generar encoding posicional financiero.

        Args:
            timestamps: lista de datetimes para cada paso (opcional)
            seq_len: longitud de secuencia si no hay timestamps

        Returns:
            (seq_len, 12) — features posicionales financieras
        """
        if timestamps is None:
            # Generar timestamps ficticios (cada hora)
            now = datetime.now(timezone.utc)
            timestamps = [now - __import__('datetime').timedelta(hours=seq_len - i) for i in range(seq_len)]

        n = len(timestamps)
        features = np.zeros((n, 12), dtype=np.float32)

        for t, ts in enumerate(timestamps):
            # 1. Día de la semana (one-hot, 5 dims — Lun a Vie)
            dow = ts.weekday()
            if dow < 5:
                features[t, dow] = 1.0

            # 2. Hora normalizada (0-1)
            features[t, 5] = ts.hour / 23.0

            # 3. Sesión de trading (one-hot, 3 dims)
            hour = ts.hour
            if 0 <= hour < 8:
                features[t, 6] = 1.0   # Asian
            elif 7 <= hour < 16:
                features[t, 7] = 1.0   # London
            if 13 <= hour < 22:
                features[t, 8] = 1.0   # NY (puede solapar con London)

            # 4. Cercanía a eventos macro (decay exponencial, 3 dims)
            day = ts.day
            features[t, 9] = np.exp(-abs(day - self.NFP_DAY) / 3.0)   # NFP proximity
            features[t, 10] = np.exp(-abs(day - self.FOMC_DAY) / 3.0)  # FOMC proximity
            features[t, 11] = np.exp(-abs(day - self.GDP_DAY) / 3.0)   # GDP proximity

        return features

    def get_sinusoidal_encoding(self, seq_len: int) -> np.ndarray:
        """
        Encoding sinusoidal estándar (Vaswani et al., 2017).
        PE(pos, 2i) = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        """
        pe = np.zeros((seq_len, self.d_model), dtype=np.float32)
        position = np.arange(seq_len).reshape(-1, 1)

        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))

        pe[:, 0::2] = np.sin(position * div_term[:self.d_model // 2 + self.d_model % 2])
        pe[:, 1::2] = np.cos(position * div_term[:self.d_model // 2])

        return pe

    def get_combined_encoding(self, timestamps: List[datetime] = None,
                               seq_len: int = 30) -> np.ndarray:
        """
        Combinación de encoding sinusoidal + financiero.
        Returns: (seq_len, d_model + 12)
        """
        sinusoidal = self.get_sinusoidal_encoding(seq_len)
        financial = self.encode(timestamps, seq_len)
        return np.concatenate([sinusoidal, financial], axis=1)


# =====================================================================
# 8. ONLINE LEARNING EXP3
# w_i(t+1) = w_i(t) × exp(η × reward_i(t)) / Z
# Auer et al. (2002)
# =====================================================================

class EXP3OnlineLearner:
    """
    Algoritmo EXP3 (Exponential-weight algorithm for Exploration and Exploitation).
    Ajusta pesos del ensemble en TIEMPO REAL basándose en rendimiento reciente.

    Diferencia con el EnsembleScorer actual:
      - EnsembleScorer: adapta lento (rolling accuracy de 20 trades)
      - EXP3: adapta RÁPIDO (cada predicción, con factor exponencial)

    El ensemble automáticamente favorece al modelo que mejor predice
    en el RÉGIMEN ACTUAL del mercado.
    """

    def __init__(
        self,
        n_experts: int = 3,
        expert_names: List[str] = None,
        eta: float = 0.1,           # Learning rate (0.05-0.2 recomendado)
        gamma: float = 0.05,         # Exploración mínima (5% a cada experto)
    ):
        self.n_experts = n_experts
        self.expert_names = expert_names or [f"expert_{i}" for i in range(n_experts)]
        self.eta = eta
        self.gamma = gamma

        # Pesos (log-scale para estabilidad numérica)
        self.log_weights = np.zeros(n_experts, dtype=np.float64)
        self.weights = np.ones(n_experts) / n_experts

        # Historial
        self.reward_history: List[Dict] = []
        self.weight_history: List[np.ndarray] = [self.weights.copy()]

    def get_weights(self) -> Dict[str, float]:
        """Obtener pesos normalizados actuales."""
        return {name: round(float(w), 4) for name, w in zip(self.expert_names, self.weights)}

    def update(self, expert_rewards: Dict[str, float]):
        """
        Actualizar pesos basándose en recompensas de cada experto.

        Args:
            expert_rewards: {"logistic_regression": 0.8, "xgboost": 0.6, "dqn_rl": 0.3}
                           (1.0 = predicción perfecta, 0.0 = incorrecta)
        """
        rewards = np.zeros(self.n_experts)
        for i, name in enumerate(self.expert_names):
            rewards[i] = expert_rewards.get(name, 0.5)

        # EXP3 update: w_i *= exp(η × reward_i / p_i)
        probs = self._get_probabilities()

        for i in range(self.n_experts):
            # Importance-weighted reward
            estimated_reward = rewards[i] / (probs[i] + 1e-10)
            self.log_weights[i] += self.eta * estimated_reward

        # Normalizar para evitar overflow
        self.log_weights -= np.max(self.log_weights)

        # Actualizar pesos normalizados
        self.weights = np.exp(self.log_weights)
        self.weights /= (self.weights.sum() + 1e-10)

        # Historial
        self.reward_history.append({
            "rewards": expert_rewards,
            "weights_after": self.get_weights(),
            "time": datetime.now(timezone.utc).isoformat(),
        })
        self.weight_history.append(self.weights.copy())

        # Trim
        if len(self.reward_history) > 200:
            self.reward_history = self.reward_history[-200:]
        if len(self.weight_history) > 200:
            self.weight_history = self.weight_history[-200:]

    def _get_probabilities(self) -> np.ndarray:
        """
        Probabilidades de selección con exploración γ.
        p_i = (1-γ) × w_i / Σw + γ / n
        """
        total = self.weights.sum() + 1e-10
        probs = (1 - self.gamma) * self.weights / total + self.gamma / self.n_experts
        return probs

    def get_ensemble_score(self, expert_predictions: Dict[str, float]) -> float:
        """
        Calcular score ponderado del ensemble usando pesos EXP3.

        Args:
            expert_predictions: {"logistic_regression": 0.72, "xgboost": 0.65, "dqn_rl": 0.8}

        Returns:
            Score ponderado (0-1)
        """
        score = 0.0
        total_weight = 0.0

        for i, name in enumerate(self.expert_names):
            if name in expert_predictions:
                score += self.weights[i] * expert_predictions[name]
                total_weight += self.weights[i]

        return score / (total_weight + 1e-10) if total_weight > 0 else 0.5

    def get_status(self) -> Dict:
        return {
            "weights": self.get_weights(),
            "eta": self.eta,
            "gamma": self.gamma,
            "n_updates": len(self.reward_history),
            "recent_rewards": self.reward_history[-5:] if self.reward_history else [],
            "weight_trend": {
                name: [round(float(wh[i]), 4) for wh in self.weight_history[-10:]]
                for i, name in enumerate(self.expert_names)
            },
        }

    def save(self, path: str = None):
        """Guardar estado."""
        path = path or "exp3_weights.json"
        try:
            state = {
                "log_weights": self.log_weights.tolist(),
                "weights": self.weights.tolist(),
                "expert_names": self.expert_names,
                "eta": self.eta,
                "gamma": self.gamma,
            }
            with open(path, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.warning(f"EXP3 save error: {e}")

    def load(self, path: str = None):
        """Cargar estado."""
        path = path or "exp3_weights.json"
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                state = json.load(f)
            self.log_weights = np.array(state["log_weights"])
            self.weights = np.array(state["weights"])
            logger.info(f"EXP3 weights loaded: {self.get_weights()}")
        except Exception as e:
            logger.warning(f"EXP3 load error: {e}")
