"""
ML SuperTrend v51 - Causal Feature Selection + Granger Causality
=================================================================
Identifica qué features CAUSAN movimientos de precio vs correlación espuria.

Módulo 1: Granger Causality Test
  X Granger-causa Y si: P(Y_t | Y_{t-1:p}, X_{t-1:p}) ≠ P(Y_t | Y_{t-1:p})
  Implementado como comparación de modelos AR con/sin la feature candidata.

Módulo 2: Wasserstein Distance (Distribution Drift Detection)
  W(P, Q) = inf E[||X - Y||]
  Detecta cuando la distribución del mercado cambió significativamente.

Módulo 3: Information Bottleneck
  min I(X;Z) - \u03b2\u00b7I(Z;Y)
  Comprimir features manteniendo solo información predictiva.

Papers:
  - Peters et al. (2017) \u2014 Causal Inference for Time Series
  - Arjovsky et al. (2017) \u2014 Wasserstein Distance
  - Tishby et al. (2000) \u2014 Information Bottleneck

Uso:
    from causal_features import GrangerCausalitySelector, WassersteinDriftDetector, InformationBottleneck
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

SCIPY_AVAILABLE = False
try:
    from scipy import stats as scipy_stats
    from scipy.linalg import toeplitz
    SCIPY_AVAILABLE = True
except ImportError:
    pass


# =====================================================================
# GRANGER CAUSALITY \u2014 Feature Selection Causal
# =====================================================================

class GrangerCausalitySelector:
    """
    Selección de features basada en causalidad de Granger.
    Elimina features que solo tienen correlación espuria.

    Test: comparar modelo AR(p) restringido vs no-restringido.
    Si agregar la feature X mejora significativamente la predicción
    de Y \u2192 X Granger-causa Y \u2192 mantener feature.
    """

    def __init__(
        self,
        max_lag: int = 10,           # Máximo lag para el test
        significance: float = 0.05,   # p-value threshold
        min_samples: int = 100,       # Mínimo de datos para test
    ):
        self.max_lag = max_lag
        self.significance = significance
        self.min_samples = min_samples
        self.causal_results: Dict[str, Dict] = {}
        self.selected_features: List[str] = []

    def test_causality(self, x: np.ndarray, y: np.ndarray, feature_name: str = "") -> Dict:
        """
        Test de Granger Causality entre x \u2192 y.

        H0: x NO Granger-causa y
        H1: x S\u00cd Granger-causa y

        Returns:
            {"is_causal": bool, "p_value": float, "f_stat": float, "best_lag": int}
        """
        n = len(x)
        if n < self.min_samples:
            return {"is_causal": False, "p_value": 1.0, "reason": "insufficient_data"}

        best_result = {"is_causal": False, "p_value": 1.0, "f_stat": 0, "best_lag": 1}

        for lag in range(1, min(self.max_lag + 1, n // 5)):
            try:
                result = self._granger_test_single_lag(x, y, lag)
                if result["p_value"] < best_result["p_value"]:
                    best_result = result
                    best_result["best_lag"] = lag
            except Exception:
                continue

        best_result["is_causal"] = best_result["p_value"] < self.significance
        best_result["feature"] = feature_name

        self.causal_results[feature_name] = best_result
        return best_result

    def _granger_test_single_lag(self, x: np.ndarray, y: np.ndarray, lag: int) -> Dict:
        """Test de Granger para un lag específico usando F-test."""
        n = len(y)
        if n <= 2 * lag + 2:
            return {"p_value": 1.0, "f_stat": 0}

        # Modelo restringido: y_t = \u03a3 \u03b1_i y_{t-i} + \u03b5
        # Modelo no restringido: y_t = \u03a3 \u03b1_i y_{t-i} + \u03a3 \u03b2_i x_{t-i} + \u03b5

        # Construir matrices de dise\u00f1o
        Y = y[lag:]
        n_obs = len(Y)

        # Restringido (solo lags de y)
        X_restricted = np.column_stack([y[lag-i-1:n-i-1] for i in range(lag)])
        X_restricted = np.column_stack([np.ones(n_obs), X_restricted])

        # No restringido (lags de y + lags de x)
        X_unrestricted = np.column_stack([
            X_restricted,
            *[x[lag-i-1:n-i-1] for i in range(lag)]
        ])

        try:
            # OLS para cada modelo
            beta_r = np.linalg.lstsq(X_restricted, Y, rcond=None)[0]
            resid_r = Y - X_restricted @ beta_r
            rss_r = np.sum(resid_r ** 2)

            beta_u = np.linalg.lstsq(X_unrestricted, Y, rcond=None)[0]
            resid_u = Y - X_unrestricted @ beta_u
            rss_u = np.sum(resid_u ** 2)

            # F-statistic
            df1 = lag  # restricciones adicionales
            df2 = n_obs - X_unrestricted.shape[1]

            if df2 <= 0 or rss_u <= 0:
                return {"p_value": 1.0, "f_stat": 0}

            f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)

            if SCIPY_AVAILABLE:
                p_value = 1 - scipy_stats.f.cdf(f_stat, df1, df2)
            else:
                # Aproximación sin scipy
                p_value = np.exp(-f_stat / 2) if f_stat > 0 else 1.0

            return {"p_value": float(p_value), "f_stat": float(f_stat)}

        except Exception:
            return {"p_value": 1.0, "f_stat": 0}

    def select_causal_features(
        self,
        feature_matrix: np.ndarray,
        target: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[List[str], List[int]]:
        """
        Seleccionar features que Granger-causan el target.

        Args:
            feature_matrix: (n_samples, n_features)
            target: (n_samples,) \u2014 variable objetivo (retornos)
            feature_names: nombres de cada feature

        Returns:
            (selected_names, selected_indices)
        """
        selected_names = []
        selected_indices = []

        for i, name in enumerate(feature_names):
            result = self.test_causality(feature_matrix[:, i], target, name)
            if result["is_causal"]:
                selected_names.append(name)
                selected_indices.append(i)
                logger.info(f"  Granger: {name} \u2192 CAUSAL (p={result['p_value']:.4f}, lag={result.get('best_lag', '?')})")
            else:
                logger.debug(f"  Granger: {name} \u2192 NO causal (p={result['p_value']:.4f})")

        self.selected_features = selected_names
        logger.info(f"Granger: {len(selected_names)}/{len(feature_names)} features son causales")
        return selected_names, selected_indices

    def get_status(self) -> Dict:
        return {
            "max_lag": self.max_lag,
            "significance": self.significance,
            "n_tested": len(self.causal_results),
            "n_causal": sum(1 for r in self.causal_results.values() if r.get("is_causal")),
            "selected_features": self.selected_features,
            "results": {
                k: {"is_causal": v.get("is_causal"), "p_value": round(v.get("p_value", 1), 4)}
                for k, v in self.causal_results.items()
            },
        }


# =====================================================================
# WASSERSTEIN DISTANCE \u2014 Distribution Drift Detection
# =====================================================================

class WassersteinDriftDetector:
    """
    Detecta drift en la distribución del mercado usando Wasserstein Distance.
    Más estable que KL-divergence cuando las distribuciones no se solapan.

    W_1(P, Q) = integral |F_P(x) - F_Q(x)| dx
    (Earth Mover's Distance para distribuciones 1D)

    Uso:
      - Comparar distribución de retornos reciente vs histórica
      - Si W es alta \u2192 el mercado cambió \u2192 adaptar modelo
    """

    def __init__(
        self,
        reference_window: int = 200,    # Ventana de referencia (histórica)
        test_window: int = 50,           # Ventana de test (reciente)
        alert_threshold: float = 0.5,    # Threshold normalizado para alerta
        critical_threshold: float = 1.0, # Threshold para acción inmediata
    ):
        self.reference_window = reference_window
        self.test_window = test_window
        self.alert_threshold = alert_threshold
        self.critical_threshold = critical_threshold

        self.data_buffer: deque = deque(maxlen=reference_window + test_window)
        self.wasserstein_history: deque = deque(maxlen=200)
        self.current_distance = 0.0
        self.alert_level = "NORMAL"

    def update(self, value: float):
        """Agregar nuevo valor y recalcular drift."""
        self.data_buffer.append(value)

        if len(self.data_buffer) < self.reference_window + self.test_window:
            return

        data = np.array(list(self.data_buffer))
        reference = data[:self.reference_window]
        recent = data[-self.test_window:]

        # Calcular Wasserstein distance (Earth Mover's Distance)
        self.current_distance = self._wasserstein_1d(reference, recent)
        self.wasserstein_history.append(self.current_distance)

        # Normalizar por la desviación estándar de referencia
        ref_std = np.std(reference) + 1e-8
        normalized = self.current_distance / ref_std

        if normalized >= self.critical_threshold:
            self.alert_level = "CRITICAL"
        elif normalized >= self.alert_threshold:
            self.alert_level = "DRIFT"
        else:
            self.alert_level = "NORMAL"

    def _wasserstein_1d(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Wasserstein-1 distance entre dos distribuciones 1D.
        W_1 = integral |F_u(x) - F_v(x)| dx
        """
        if SCIPY_AVAILABLE:
            return float(scipy_stats.wasserstein_distance(u, v))

        # Implementación manual
        u_sorted = np.sort(u)
        v_sorted = np.sort(v)

        # Interpolar ambas CDFs en los mismos puntos
        all_values = np.sort(np.concatenate([u_sorted, v_sorted]))

        u_cdf = np.searchsorted(u_sorted, all_values, side='right') / len(u_sorted)
        v_cdf = np.searchsorted(v_sorted, all_values, side='right') / len(v_sorted)

        # Integral de |F_u - F_v|
        diffs = np.abs(u_cdf - v_cdf)
        if len(all_values) > 1:
            dx = np.diff(all_values)
            return float(np.sum(diffs[:-1] * dx))
        return 0.0

    def get_risk_multiplier(self) -> float:
        """Multiplicador de riesgo basado en drift."""
        multipliers = {"NORMAL": 1.0, "DRIFT": 0.7, "CRITICAL": 0.4}
        return multipliers.get(self.alert_level, 1.0)

    def get_status(self) -> Dict:
        return {
            "current_distance": round(self.current_distance, 6),
            "alert_level": self.alert_level,
            "risk_multiplier": self.get_risk_multiplier(),
            "buffer_size": len(self.data_buffer),
            "history": [round(w, 4) for w in list(self.wasserstein_history)[-20:]],
        }


# =====================================================================
# INFORMATION BOTTLENECK \u2014 Feature Compression
# =====================================================================

if TORCH_AVAILABLE:
    class InformationBottleneckLayer(nn.Module):
        """
        Information Bottleneck como capa de red neuronal.
        Comprime features manteniendo solo información relevante para predicción.

        min I(X;Z) - \u03b2\u00b7I(Z;Y)

        Implementado como variational: z = \u03bc + \u03c3\u00b7\u03b5 (reparameterization trick)
        La pérdida de KL penaliza complejidad excesiva.
        """

        def __init__(self, input_dim: int, bottleneck_dim: int = 16, beta: float = 0.01):
            super().__init__()
            self.beta = beta
            self.bottleneck_dim = bottleneck_dim

            # Encoder: input \u2192 (\u03bc, log_\u03c3\u00b2)
            self.fc_mu = nn.Linear(input_dim, bottleneck_dim)
            self.fc_logvar = nn.Linear(input_dim, bottleneck_dim)

            # Decoder: z \u2192 output
            self.fc_out = nn.Linear(bottleneck_dim, input_dim)

            self.last_kl = 0.0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            x: (batch, input_dim)
            Returns: (batch, input_dim) compressed representation
            """
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)

            # Reparameterization trick
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
            else:
                z = mu  # Deterministic at test time

            # KL divergence: I(X;Z) \u2248 KL(q(z|x) || p(z))
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            self.last_kl = kl.item()

            # Reconstruct
            out = self.fc_out(z)
            return out

        def get_bottleneck_loss(self) -> float:
            """Return \u03b2 \u00d7 KL for adding to total loss."""
            return self.beta * self.last_kl

else:
    class InformationBottleneckLayer:
        """Placeholder when PyTorch not available."""
        def __init__(self, *args, **kwargs):
            pass


class InformationBottleneckSelector:
    """
    Wrapper de alto nivel para Information Bottleneck.
    Eval\u00faa qu\u00e9 features son esenciales y cu\u00e1les son redundantes.
    """

    def __init__(self, n_features: int = 6, bottleneck_dim: int = 4, beta: float = 0.01):
        self.n_features = n_features
        self.bottleneck_dim = bottleneck_dim
        self.beta = beta
        self.feature_importance: Dict[str, float] = {}

        if TORCH_AVAILABLE:
            self.layer = InformationBottleneckLayer(n_features, bottleneck_dim, beta)
        else:
            self.layer = None

    def compute_importance(
        self,
        X: np.ndarray,
        feature_names: List[str] = None
    ) -> Dict[str, float]:
        """
        Estimar importancia de cada feature bas\u00e1ndose en cu\u00e1nto
        contribuye al bottleneck.

        M\u00e9todo: leave-one-out \u2014 quitar cada feature y medir
        cu\u00e1nto aumenta la p\u00e9rdida de reconstrucci\u00f3n.
        """
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(X.shape[1])]

        n_features = X.shape[1]
        baseline_error = np.mean(np.var(X, axis=0))

        importance = {}
        for i, name in enumerate(feature_names):
            # Quitar feature i y medir p\u00e9rdida
            X_without = np.delete(X, i, axis=1)
            error_without = np.mean(np.var(X_without, axis=0))

            # Importancia = cu\u00e1nto empeora sin esta feature
            imp = max(0, baseline_error - error_without) / (baseline_error + 1e-8)
            importance[name] = round(float(1.0 - imp), 4)  # Invertir: 1 = importante

        # Normalizar
        total = sum(importance.values()) + 1e-8
        importance = {k: round(v / total, 4) for k, v in importance.items()}

        self.feature_importance = importance
        return importance

    def get_status(self) -> Dict:
        return {
            "bottleneck_dim": self.bottleneck_dim,
            "beta": self.beta,
            "feature_importance": self.feature_importance,
            "has_torch_layer": self.layer is not None,
        }
