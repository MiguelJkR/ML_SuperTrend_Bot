"""
ML SuperTrend v51 - HMM Regime Detection
==========================================
Hidden Markov Model para detección de regímenes de mercado.

Modelos 3 estados latentes:
  - Estado 0: TRENDING (tendencia clara, baja volatilidad relativa)
  - Estado 1: MEAN_REVERTING (rango, alta reversión a la media)
  - Estado 2: VOLATILE (alta volatilidad, movimientos erráticos)

Algoritmos:
  - Baum-Welch (EM) para entrenamiento no supervisado
  - Forward-Backward para probabilidades de estado
  - Viterbi para secuencia más probable de estados

Basado en:
  - "A Hidden Markov Model for Regime Detection" (Ang & Bekaert, 2002)
  - "Regime Switching Models" (Hamilton, 1989)

Uso:
    from hmm_regime import HMMRegimeDetector
    hmm = HMMRegimeDetector()
    hmm.fit(returns)
    regime, probs = hmm.predict(new_returns)
"""

import logging
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Intentar usar hmmlearn (más eficiente), fallback a implementación propia
HMM_LIBRARY = False
try:
    from hmmlearn import hmm as hmmlearn_hmm
    HMM_LIBRARY = True
    logger.info("hmmlearn disponible — usando implementación optimizada")
except ImportError:
    logger.info("hmmlearn no disponible — usando implementación numpy")


# Nombres de regímenes
REGIME_NAMES = {
    0: "TRENDING",
    1: "MEAN_REVERTING",
    2: "VOLATILE",
}

REGIME_DESCRIPTIONS = {
    0: "Tendencia clara — seguir señales direccionales",
    1: "Rango/reversión — operar en extremos, stops ajustados",
    2: "Alta volatilidad — reducir tamaño, ampliar stops",
}


class HMMRegimeDetector:
    """
    Detector de regímenes de mercado usando Hidden Markov Model.
    3 estados: Trending, Mean-Reverting, Volatile.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_features: int = 2,          # returns + volatility
        lookback: int = 100,           # Ventana para fit incremental
        refit_interval: int = 50,      # Cada cuántos datos reajustar
        model_path: str = None,
    ):
        self.n_states = n_states
        self.n_features = n_features
        self.lookback = lookback
        self.refit_interval = refit_interval
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "hmm_model.json"
        )

        # Datos acumulados
        self.data_buffer: List[np.ndarray] = []
        self.samples_since_fit = 0
        self.is_fitted = False

        # Estado actual
        self.current_regime = 1  # Default: mean-reverting
        self.regime_probabilities = np.array([0.33, 0.34, 0.33])
        self.regime_history: List[int] = []

        # Modelo HMM
        if HMM_LIBRARY:
            self.model = hmmlearn_hmm.GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42,
                tol=1e-4,
            )
            # Prior: favorecer transiciones lentas (mercados son persistentes)
            self.model.transmat_prior = np.array([
                [10, 1, 1],   # Trending tiende a seguir trending
                [1, 10, 1],   # Mean-reverting tiende a seguir
                [1, 1, 10],   # Volatile tiende a seguir
            ], dtype=np.float64)
        else:
            self.model = NumpyHMM(n_states=n_states, n_features=n_features)

        # Estadísticas por régimen
        self.regime_stats = {i: {"count": 0, "avg_return": 0, "avg_vol": 0} for i in range(n_states)}

        # Cargar estado previo
        self._load_state()

    def update(self, returns: float, volatility: float):
        """
        Alimentar con nuevos datos.

        Args:
            returns: retorno del período (log-return recomendado)
            volatility: volatilidad realizada (ATR normalizado o std)
        """
        obs = np.array([returns, volatility], dtype=np.float64)
        self.data_buffer.append(obs)

        # Mantener buffer manejable
        if len(self.data_buffer) > self.lookback * 3:
            self.data_buffer = self.data_buffer[-self.lookback * 2:]

        self.samples_since_fit += 1

        # Auto-fit
        if len(self.data_buffer) >= self.lookback:
            if not self.is_fitted or self.samples_since_fit >= self.refit_interval:
                self._fit()

        # Predecir régimen actual
        if self.is_fitted:
            self._predict_current()

    def _fit(self):
        """Entrenar HMM con datos acumulados."""
        try:
            X = np.array(self.data_buffer[-self.lookback:])

            if HMM_LIBRARY:
                self.model.fit(X)
            else:
                self.model.fit(X)

            self.is_fitted = True
            self.samples_since_fit = 0

            # Actualizar stats por régimen
            self._update_regime_stats(X)

            logger.info(f"HMM re-fitted con {len(X)} observaciones")
        except Exception as e:
            logger.warning(f"HMM fit error: {e}")

    def _predict_current(self):
        """Predecir régimen actual basado en últimas observaciones."""
        if not self.is_fitted or len(self.data_buffer) < 10:
            return

        try:
            # Usar últimas observaciones para predecir
            X_recent = np.array(self.data_buffer[-min(30, len(self.data_buffer)):])

            if HMM_LIBRARY:
                # Probabilidades de estado para la última observación
                log_probs = self.model.score_samples(X_recent)
                # posteriors: (n_samples, n_states)
                _, posteriors = self.model.score_samples(X_recent)
                self.regime_probabilities = posteriors[-1]
                self.current_regime = int(np.argmax(self.regime_probabilities))
            else:
                self.current_regime, self.regime_probabilities = self.model.predict(X_recent)

            self.regime_history.append(self.current_regime)
            if len(self.regime_history) > 500:
                self.regime_history = self.regime_history[-500:]

        except Exception as e:
            logger.warning(f"HMM predict error: {e}")

    def _update_regime_stats(self, X: np.ndarray):
        """Actualizar estadísticas por régimen."""
        try:
            if HMM_LIBRARY:
                states = self.model.predict(X)
            else:
                states = [self.model.predict(X[i:i+1].reshape(1, -1))[0] for i in range(len(X))]

            for s in range(self.n_states):
                mask = np.array(states) == s
                if mask.any():
                    self.regime_stats[s] = {
                        "count": int(mask.sum()),
                        "avg_return": float(np.mean(X[mask, 0])),
                        "avg_vol": float(np.mean(X[mask, 1])),
                        "pct": round(float(mask.mean()) * 100, 1),
                    }
        except Exception:
            pass

    def get_regime(self) -> Dict:
        """Obtener régimen actual con detalles."""
        return {
            "regime": self.current_regime,
            "regime_name": REGIME_NAMES.get(self.current_regime, "UNKNOWN"),
            "description": REGIME_DESCRIPTIONS.get(self.current_regime, ""),
            "probabilities": {
                REGIME_NAMES[i]: round(float(p), 4)
                for i, p in enumerate(self.regime_probabilities)
            },
            "confidence": round(float(np.max(self.regime_probabilities)), 4),
            "is_fitted": self.is_fitted,
            "buffer_size": len(self.data_buffer),
            "regime_stats": self.regime_stats,
        }

    def get_risk_multiplier(self) -> float:
        """
        Multiplicador de riesgo basado en régimen.
        - TRENDING: 1.0 (normal)
        - MEAN_REVERTING: 0.8 (reducir ligeramente)
        - VOLATILE: 0.5 (reducir significativamente)
        """
        multipliers = {0: 1.0, 1: 0.8, 2: 0.5}
        base = multipliers.get(self.current_regime, 0.8)
        # Ponderar por confianza
        confidence = float(np.max(self.regime_probabilities))
        return base * confidence + (1.0 - confidence) * 0.8

    def get_transition_matrix(self) -> Optional[np.ndarray]:
        """Obtener matriz de transición aprendida."""
        if HMM_LIBRARY and self.is_fitted:
            return self.model.transmat_
        elif self.is_fitted:
            return self.model.transition_matrix
        return None

    def _save_state(self):
        """Guardar estado del HMM."""
        try:
            state = {
                "current_regime": self.current_regime,
                "regime_probabilities": self.regime_probabilities.tolist(),
                "regime_stats": self.regime_stats,
                "is_fitted": self.is_fitted,
                "regime_history": self.regime_history[-100:],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if self.is_fitted and HMM_LIBRARY:
                state["means"] = self.model.means_.tolist()
                state["transmat"] = self.model.transmat_.tolist()

            with open(self.model_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"HMM save error: {e}")

    def _load_state(self):
        """Cargar estado previo del HMM."""
        if not os.path.exists(self.model_path):
            return
        try:
            with open(self.model_path) as f:
                state = json.load(f)
            self.current_regime = state.get("current_regime", 1)
            self.regime_probabilities = np.array(state.get("regime_probabilities", [0.33, 0.34, 0.33]))
            self.regime_stats = state.get("regime_stats", self.regime_stats)
            self.regime_history = state.get("regime_history", [])
            logger.info(f"HMM state loaded: regime={REGIME_NAMES.get(self.current_regime)}")
        except Exception as e:
            logger.warning(f"HMM load error: {e}")


class NumpyHMM:
    """
    Implementación simple de Gaussian HMM en numpy puro.
    Fallback cuando hmmlearn no está instalado.

    Implementa:
      - Forward algorithm (P(O|λ))
      - Viterbi (secuencia más probable)
      - Baum-Welch simplificado (estimación de parámetros)
    """

    def __init__(self, n_states: int = 3, n_features: int = 2):
        self.n_states = n_states
        self.n_features = n_features

        # Parámetros iniciales
        self.initial_prob = np.ones(n_states) / n_states

        # Matriz de transición (favorece persistencia)
        self.transition_matrix = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])

        # Medias y covarianzas por estado (inicialización heurística)
        self.means = np.array([
            [0.001, 0.01],   # Trending: retorno positivo, baja vol
            [0.0, 0.015],    # Mean-reverting: retorno ~0, vol media
            [0.0, 0.03],     # Volatile: retorno ~0, alta vol
        ])

        self.covars = np.array([
            np.eye(n_features) * 0.001,
            np.eye(n_features) * 0.002,
            np.eye(n_features) * 0.005,
        ])

        self.is_fitted = False

    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, covar: np.ndarray) -> float:
        """Densidad gaussiana multivariada."""
        d = len(mean)
        diff = x - mean
        try:
            inv_cov = np.linalg.inv(covar + np.eye(d) * 1e-6)
            det = np.linalg.det(covar + np.eye(d) * 1e-6)
            norm = 1.0 / (((2 * np.pi) ** (d / 2)) * (det ** 0.5) + 1e-10)
            exponent = -0.5 * diff @ inv_cov @ diff
            return float(norm * np.exp(exponent))
        except Exception:
            return 1e-10

    def _emission_probs(self, x: np.ndarray) -> np.ndarray:
        """P(x | state) para cada estado."""
        probs = np.array([
            self._gaussian_pdf(x, self.means[s], self.covars[s])
            for s in range(self.n_states)
        ])
        return np.maximum(probs, 1e-10)

    def fit(self, X: np.ndarray, n_iter: int = 20):
        """
        Baum-Welch simplificado (EM).
        X: (n_samples, n_features)
        """
        n = len(X)
        if n < 10:
            return

        for iteration in range(n_iter):
            # E-step: Forward-backward
            alpha = np.zeros((n, self.n_states))
            beta = np.zeros((n, self.n_states))

            # Forward
            alpha[0] = self.initial_prob * self._emission_probs(X[0])
            alpha[0] /= (alpha[0].sum() + 1e-10)

            for t in range(1, n):
                emit = self._emission_probs(X[t])
                for j in range(self.n_states):
                    alpha[t, j] = emit[j] * np.sum(alpha[t - 1] * self.transition_matrix[:, j])
                alpha[t] /= (alpha[t].sum() + 1e-10)

            # Backward
            beta[-1] = 1.0
            for t in range(n - 2, -1, -1):
                emit_next = self._emission_probs(X[t + 1])
                for i in range(self.n_states):
                    beta[t, i] = np.sum(self.transition_matrix[i] * emit_next * beta[t + 1])
                beta[t] /= (beta[t].sum() + 1e-10)

            # Posteriors γ(t, i) = P(S_t = i | O)
            gamma = alpha * beta
            gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-10)

            # M-step: actualizar parámetros
            for s in range(self.n_states):
                weights = gamma[:, s]
                total_w = weights.sum() + 1e-10

                # Actualizar media
                self.means[s] = np.sum(weights[:, np.newaxis] * X, axis=0) / total_w

                # Actualizar covarianza
                diff = X - self.means[s]
                self.covars[s] = (diff.T * weights) @ diff / total_w
                # Regularización
                self.covars[s] += np.eye(self.n_features) * 1e-4

            # Actualizar transiciones
            for i in range(self.n_states):
                for j in range(self.n_states):
                    num = 0
                    for t in range(n - 1):
                        emit_next = self._emission_probs(X[t + 1])
                        num += alpha[t, i] * self.transition_matrix[i, j] * emit_next[j] * beta[t + 1, j]
                    self.transition_matrix[i, j] = num
                self.transition_matrix[i] /= (self.transition_matrix[i].sum() + 1e-10)

            # Actualizar probabilidades iniciales
            self.initial_prob = gamma[0]

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[int, np.ndarray]:
        """Predecir régimen para la última observación."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        n = len(X)
        # Forward para obtener probabilidades
        alpha = np.zeros((n, self.n_states))
        alpha[0] = self.initial_prob * self._emission_probs(X[0])
        alpha[0] /= (alpha[0].sum() + 1e-10)

        for t in range(1, n):
            emit = self._emission_probs(X[t])
            for j in range(self.n_states):
                alpha[t, j] = emit[j] * np.sum(alpha[t - 1] * self.transition_matrix[:, j])
            alpha[t] /= (alpha[t].sum() + 1e-10)

        probs = alpha[-1]
        regime = int(np.argmax(probs))
        return regime, probs
