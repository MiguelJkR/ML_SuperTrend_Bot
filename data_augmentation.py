"""
ML SuperTrend v51 - Data Augmentation para Series Temporales
=============================================================
Técnicas científicas para multiplicar datos de entrenamiento x5-10
sin necesidad de datos nuevos.

Basado en:
  - "Data Augmentation for Time Series Classification" (Um et al., 2017)
  - "Time Series Data Augmentation for Deep Learning" (Wen et al., 2021)

Técnicas implementadas:
  1. Jittering: Ruido gaussiano controlado
  2. Scaling (Magnitude Warping): Distorsión de amplitud con splines
  3. Time Warping: Distorsión temporal con splines cúbicos
  4. Window Slicing: Sub-secuencias aleatorias del buffer
  5. Permutation: Segmentos permutados (rompe dependencia temporal parcial)
  6. Mixup: Combinación convexa de dos secuencias (Zhang et al., 2017)

Uso:
    from data_augmentation import TimeSeriesAugmentor
    aug = TimeSeriesAugmentor()
    X_aug, y_aug = aug.augment(X_train, y_train, factor=5)
"""

import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class TimeSeriesAugmentor:
    """
    Aumenta datos de series temporales para entrenamiento de LSTM/RNN.
    Diseñado para datos financieros (candles normalizados).
    """

    def __init__(
        self,
        jitter_sigma: float = 0.03,       # σ para ruido gaussiano
        scale_sigma: float = 0.1,          # σ para magnitude warping
        warp_sigma: float = 0.2,           # σ para time warping
        warp_knots: int = 4,               # Puntos de control del spline
        window_ratio: float = 0.9,         # Ratio de ventana para slicing
        n_segments: int = 5,               # Segmentos para permutation
        mixup_alpha: float = 0.2,          # Alpha para distribución Beta en mixup
        seed: int = None,
    ):
        self.jitter_sigma = jitter_sigma
        self.scale_sigma = scale_sigma
        self.warp_sigma = warp_sigma
        self.warp_knots = warp_knots
        self.window_ratio = window_ratio
        self.n_segments = n_segments
        self.mixup_alpha = mixup_alpha
        self.rng = np.random.RandomState(seed)

    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        factor: int = 5,
        techniques: List[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generar datos aumentados.

        Args:
            X: (n_samples, seq_len, n_features) — secuencias originales
            y: (n_samples,) — labels originales
            factor: cuántas copias aumentadas generar (total = original + factor)
            techniques: lista de técnicas a usar. Default: todas.

        Returns:
            X_aug: (n_samples * (1 + factor), seq_len, n_features)
            y_aug: (n_samples * (1 + factor),)
        """
        if techniques is None:
            techniques = ["jitter", "scaling", "time_warp", "window_slice", "permutation", "mixup"]

        all_X = [X]
        all_y = [y]
        n_per_technique = max(1, factor // len(techniques))

        for tech in techniques:
            for _ in range(n_per_technique):
                try:
                    if tech == "jitter":
                        X_new = self._jitter(X)
                        y_new = y.copy()
                    elif tech == "scaling":
                        X_new = self._magnitude_warp(X)
                        y_new = y.copy()
                    elif tech == "time_warp":
                        X_new = self._time_warp(X)
                        y_new = y.copy()
                    elif tech == "window_slice":
                        X_new, y_new = self._window_slice(X, y)
                    elif tech == "permutation":
                        X_new = self._permutation(X)
                        y_new = y.copy()
                    elif tech == "mixup":
                        X_new, y_new = self._mixup(X, y)
                    else:
                        continue

                    all_X.append(X_new)
                    all_y.append(y_new)
                except Exception as e:
                    logger.warning(f"Augmentation '{tech}' error: {e}")

        X_aug = np.concatenate(all_X, axis=0)
        y_aug = np.concatenate(all_y, axis=0)

        # Shuffle
        indices = self.rng.permutation(len(X_aug))
        X_aug = X_aug[indices]
        y_aug = y_aug[indices]

        logger.info(f"Data Augmentation: {len(X)} → {len(X_aug)} secuencias "
                    f"(x{len(X_aug)/len(X):.1f}) | técnicas: {techniques}")

        return X_aug, y_aug

    # ──────────────────────────────────────────────
    # TÉCNICA 1: Jittering (ruido gaussiano)
    # x' = x + N(0, σ)
    # ──────────────────────────────────────────────
    def _jitter(self, X: np.ndarray) -> np.ndarray:
        """Agregar ruido gaussiano controlado a cada feature."""
        noise = self.rng.normal(0, self.jitter_sigma, X.shape).astype(np.float32)
        return X + noise

    # ──────────────────────────────────────────────
    # TÉCNICA 2: Magnitude Warping (distorsión de amplitud)
    # x' = x × cubic_spline(random_knots)
    # ──────────────────────────────────────────────
    def _magnitude_warp(self, X: np.ndarray) -> np.ndarray:
        """Escalar la magnitud usando splines cúbicos aleatorios."""
        n_samples, seq_len, n_features = X.shape
        X_new = np.zeros_like(X)

        for i in range(n_samples):
            # Generar curva de escalado suave con spline
            warp_curve = self._generate_smooth_curve(seq_len, self.scale_sigma)
            # Aplicar a todas las features
            for f in range(n_features):
                X_new[i, :, f] = X[i, :, f] * warp_curve

        return X_new

    # ──────────────────────────────────────────────
    # TÉCNICA 3: Time Warping (distorsión temporal)
    # Deforma el eje temporal con splines cúbicos
    # ──────────────────────────────────────────────
    def _time_warp(self, X: np.ndarray) -> np.ndarray:
        """Distorsionar el eje temporal usando splines."""
        n_samples, seq_len, n_features = X.shape
        X_new = np.zeros_like(X)

        for i in range(n_samples):
            # Generar mapping temporal distorsionado
            warp_curve = self._generate_smooth_curve(seq_len, self.warp_sigma)
            # Convertir a indices temporales
            time_orig = np.arange(seq_len, dtype=np.float64)
            time_warped = np.cumsum(warp_curve)
            time_warped = time_warped / time_warped[-1] * (seq_len - 1)  # Normalizar

            # Interpolar cada feature
            for f in range(n_features):
                X_new[i, :, f] = np.interp(time_orig, time_warped, X[i, :, f])

        return X_new

    # ──────────────────────────────────────────────
    # TÉCNICA 4: Window Slicing
    # Sub-secuencias aleatorias del buffer
    # ──────────────────────────────────────────────
    def _window_slice(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tomar sub-ventanas aleatorias y redimensionar al tamaño original."""
        n_samples, seq_len, n_features = X.shape
        window_size = max(int(seq_len * self.window_ratio), 2)
        X_new = np.zeros_like(X)

        for i in range(n_samples):
            start = self.rng.randint(0, seq_len - window_size + 1)
            window = X[i, start:start + window_size, :]

            # Redimensionar al tamaño original usando interpolación
            for f in range(n_features):
                X_new[i, :, f] = np.interp(
                    np.linspace(0, 1, seq_len),
                    np.linspace(0, 1, window_size),
                    window[:, f]
                )

        return X_new, y.copy()

    # ──────────────────────────────────────────────
    # TÉCNICA 5: Permutation
    # Segmentos permutados (rompe dependencia parcial)
    # ──────────────────────────────────────────────
    def _permutation(self, X: np.ndarray) -> np.ndarray:
        """Permutar segmentos de la secuencia."""
        n_samples, seq_len, n_features = X.shape
        X_new = np.zeros_like(X)
        seg_len = max(seq_len // self.n_segments, 1)

        for i in range(n_samples):
            # Dividir en segmentos
            segments = []
            for s in range(0, seq_len, seg_len):
                end = min(s + seg_len, seq_len)
                segments.append(X[i, s:end, :])

            # Permutar
            self.rng.shuffle(segments)

            # Reconstruir
            idx = 0
            for seg in segments:
                end = idx + len(seg)
                if end > seq_len:
                    end = seq_len
                    seg = seg[:end - idx]
                X_new[i, idx:end, :] = seg
                idx = end

        return X_new

    # ──────────────────────────────────────────────
    # TÉCNICA 6: Mixup
    # x' = λ·x_i + (1-λ)·x_j, y' = λ·y_i + (1-λ)·y_j
    # Zhang et al., 2017
    # ──────────────────────────────────────────────
    def _mixup(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Combinación convexa de pares de secuencias."""
        n_samples = len(X)
        X_new = np.zeros_like(X)
        y_new = np.zeros_like(y, dtype=np.float32)

        # Generar pares aleatorios
        indices = self.rng.permutation(n_samples)

        for i in range(n_samples):
            j = indices[i]
            lam = self.rng.beta(self.mixup_alpha, self.mixup_alpha)
            X_new[i] = lam * X[i] + (1 - lam) * X[j]
            y_new[i] = lam * y[i] + (1 - lam) * y[j]

        return X_new, y_new

    # ──────────────────────────────────────────────
    # UTILIDADES
    # ──────────────────────────────────────────────
    def _generate_smooth_curve(self, length: int, sigma: float) -> np.ndarray:
        """Generar una curva suave alrededor de 1.0 usando interpolación de puntos aleatorios."""
        # Puntos de control aleatorios
        knot_positions = np.linspace(0, length - 1, self.warp_knots + 2)
        knot_values = self.rng.normal(1.0, sigma, self.warp_knots + 2)
        # Asegurar que empieza y termina cerca de 1.0
        knot_values[0] = 1.0 + self.rng.normal(0, sigma * 0.3)
        knot_values[-1] = 1.0 + self.rng.normal(0, sigma * 0.3)

        # Interpolar suavemente
        positions = np.arange(length, dtype=np.float64)
        curve = np.interp(positions, knot_positions, knot_values)

        # Asegurar valores positivos
        curve = np.maximum(curve, 0.1)

        return curve.astype(np.float32)

    def get_status(self) -> dict:
        """Estado del augmentor para dashboard."""
        return {
            "jitter_sigma": self.jitter_sigma,
            "scale_sigma": self.scale_sigma,
            "warp_sigma": self.warp_sigma,
            "window_ratio": self.window_ratio,
            "mixup_alpha": self.mixup_alpha,
            "techniques": ["jitter", "scaling", "time_warp", "window_slice", "permutation", "mixup"],
        }
