"""
ML SuperTrend v51 - Kelly Criterion + Dynamic Position Sizing
================================================================
Calcula el tamaño óptimo de posición usando el criterio de Kelly,
ajustado dinámicamente por régimen HMM, incertidumbre MC Dropout,
drift de Wasserstein y volatilidad wavelet.

Fórmulas:
  Kelly clásico:
    f* = (p × b - q) / b
    donde p = prob(win), b = avg_win/avg_loss, q = 1-p

  Half-Kelly (más conservador):
    f_half = f* / 2

  Ajustes dinámicos:
    f_adjusted = f_half × regime_mult × uncertainty_mult × drift_mult × vol_mult

  Regime multipliers (HMM):
    TRENDING     → 1.0  (full size)
    MEAN_REVERT  → 0.7  (reducido)
    VOLATILE     → 0.4  (muy reducido)

  Uncertainty multiplier (MC Dropout):
    should_trade=True  → 1.0
    should_trade=False → 0.3
    Interpolado por std: mult = max(0.3, 1 - std/threshold)

  Drift multiplier (Wasserstein):
    NORMAL   → 1.0
    DRIFT    → 0.7
    CRITICAL → 0.4

Papers:
  - Kelly (1956) — "A New Interpretation of Information Rate"
  - Thorp (2006) — "The Kelly Criterion in Blackjack, Sports Betting and the Stock Market"
  - Vince (1992) — "The Mathematics of Money Management"

Uso:
    from kelly_sizing import KellySizer
    ks = KellySizer()
    ks.update_stats(trades)
    size = ks.get_position_size(capital=10000, trader=trader)
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class KellySizer:
    """
    Position sizing dinámico basado en Kelly Criterion con
    ajustes por régimen, incertidumbre y drift.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,      # Half-Kelly por defecto (más conservador)
        min_size_pct: float = 0.25,        # Mínimo 0.25% del capital
        max_size_pct: float = 3.0,         # Máximo 3% del capital
        lookback_trades: int = 50,         # Trades para calcular stats
        min_trades: int = 10,              # Mínimo de trades para Kelly
        default_size_pct: float = 1.0,     # Default si no hay suficientes datos
    ):
        self.kelly_fraction = kelly_fraction
        self.min_size_pct = min_size_pct
        self.max_size_pct = max_size_pct
        self.lookback_trades = lookback_trades
        self.min_trades = min_trades
        self.default_size_pct = default_size_pct

        # Trade stats
        self.trade_history: deque = deque(maxlen=lookback_trades)
        self.win_rate: float = 0.5
        self.avg_win: float = 0.0
        self.avg_loss: float = 0.0
        self.payoff_ratio: float = 1.0

        # Kelly results
        self.raw_kelly: float = 0.0
        self.adjusted_kelly: float = 0.0
        self.last_adjustments: Dict[str, float] = {}
        self.last_size_pct: float = default_size_pct

    def update_stats(self, trades: List[Dict]):
        """
        Actualizar estadísticas de trading para el cálculo de Kelly.

        Args:
            trades: Lista de dicts con al menos 'pnl' key
        """
        for t in trades:
            pnl = t.get('pnl', 0)
            self.trade_history.append(pnl)

        if len(self.trade_history) < self.min_trades:
            return

        pnls = np.array(list(self.trade_history))
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        self.win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0.5
        self.avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
        self.avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 1

        self.payoff_ratio = self.avg_win / max(self.avg_loss, 0.01)

        # Kelly Criterion: f* = (p*b - q) / b
        p = self.win_rate
        b = self.payoff_ratio
        q = 1 - p

        self.raw_kelly = (p * b - q) / b if b > 0 else 0
        self.raw_kelly = max(0, self.raw_kelly)  # No posiciones cortas de capital

        logger.debug(f"Kelly stats: p={p:.2%}, b={b:.2f}, f*={self.raw_kelly:.4f}")

    def get_position_size(
        self,
        capital: float,
        trader=None,
        signal_confidence: float = 1.0,
    ) -> Dict:
        """
        Calcular tamaño de posición óptimo ajustado.

        Args:
            capital: Capital disponible
            trader: Instancia del trader (para acceder a módulos)
            signal_confidence: Confianza de la señal (0-1)

        Returns:
            {
                "size_pct": float,          # % del capital a arriesgar
                "size_units": float,         # Unidades monetarias
                "kelly_raw": float,          # Kelly puro
                "kelly_adjusted": float,     # Kelly ajustado
                "adjustments": dict,         # Multiplicadores aplicados
            }
        """
        # Base Kelly
        if len(self.trade_history) >= self.min_trades:
            base_kelly = self.raw_kelly * self.kelly_fraction
        else:
            base_kelly = self.default_size_pct / 100

        # ── Ajustes dinámicos ──
        adjustments = {}

        # 1. Régimen HMM
        regime_mult = self._get_regime_multiplier(trader)
        adjustments["regime"] = regime_mult

        # 2. MC Dropout uncertainty
        uncertainty_mult = self._get_uncertainty_multiplier(trader)
        adjustments["uncertainty"] = uncertainty_mult

        # 3. Wasserstein drift
        drift_mult = self._get_drift_multiplier(trader)
        adjustments["drift"] = drift_mult

        # 4. Wavelet noise level
        vol_mult = self._get_wavelet_multiplier(trader)
        adjustments["wavelet_noise"] = vol_mult

        # 5. Signal confidence
        conf_mult = 0.5 + signal_confidence * 0.5  # [0.5, 1.0]
        adjustments["signal_confidence"] = round(conf_mult, 3)

        # 6. Drawdown protection
        dd_mult = self._get_drawdown_multiplier(trader)
        adjustments["drawdown"] = dd_mult

        # Combinar
        total_mult = (regime_mult * uncertainty_mult * drift_mult *
                      vol_mult * conf_mult * dd_mult)

        adjusted_kelly = base_kelly * total_mult

        # Clamp to min/max
        size_pct = np.clip(adjusted_kelly * 100, self.min_size_pct, self.max_size_pct)
        size_units = capital * (size_pct / 100)

        self.adjusted_kelly = adjusted_kelly
        self.last_adjustments = adjustments
        self.last_size_pct = size_pct

        return {
            "size_pct": round(float(size_pct), 3),
            "size_units": round(float(size_units), 2),
            "kelly_raw": round(float(self.raw_kelly), 4),
            "kelly_adjusted": round(float(adjusted_kelly), 4),
            "adjustments": adjustments,
            "base_kelly_fraction": self.kelly_fraction,
        }

    def _get_regime_multiplier(self, trader) -> float:
        """Multiplicador basado en régimen HMM."""
        hmm = getattr(trader, 'hmm_regime', None) if trader else None
        if not hmm:
            return 1.0

        try:
            regime_info = hmm.get_regime()
            regime = regime_info.get("regime", "unknown")
            multipliers = {
                "TRENDING": 1.0,
                "MEAN_REVERTING": 0.7,
                "VOLATILE": 0.4,
            }
            return multipliers.get(regime, 0.8)
        except Exception:
            return 1.0

    def _get_uncertainty_multiplier(self, trader) -> float:
        """Multiplicador basado en MC Dropout uncertainty."""
        lstm = getattr(trader, 'lstm_predictor', None) if trader else None
        if not lstm:
            return 1.0

        uncertainty = getattr(lstm, 'last_uncertainty', None)
        if not uncertainty or not isinstance(uncertainty, dict):
            return 1.0

        if not uncertainty.get("should_trade", True):
            return 0.3

        # Interpolate: lower std = higher multiplier
        std = uncertainty.get("std", 0)
        threshold = uncertainty.get("threshold", 0.15)
        if threshold > 0:
            mult = max(0.3, 1.0 - (std / threshold) * 0.7)
            return round(mult, 3)

        return 1.0

    def _get_drift_multiplier(self, trader) -> float:
        """Multiplicador basado en Wasserstein drift."""
        wd = getattr(trader, 'wasserstein_drift', None) if trader else None
        if wd:
            return wd.get_risk_multiplier()

        # Try from LSTM
        lstm = getattr(trader, 'lstm_predictor', None) if trader else None
        if lstm:
            drift = getattr(lstm, 'last_drift_status', None)
            if drift and isinstance(drift, dict):
                return drift.get("risk_multiplier", 1.0)

        return 1.0

    def _get_wavelet_multiplier(self, trader) -> float:
        """Multiplicador basado en nivel de ruido wavelet."""
        lstm = getattr(trader, 'lstm_predictor', None) if trader else None
        if not lstm:
            return 1.0

        energy = getattr(lstm, 'last_wavelet_energy', None)
        if not energy or not isinstance(energy, dict):
            return 1.0

        # Check noise percentage (level 4 = high frequency noise)
        noise = energy.get("level_4", {})
        noise_pct = noise.get("pct", 0)

        if noise_pct > 50:
            return 0.5   # Very noisy — half size
        elif noise_pct > 35:
            return 0.7   # Noisy — reduced
        else:
            return 1.0   # Normal

    def _get_drawdown_multiplier(self, trader) -> float:
        """Reducir tamaño progresivamente durante drawdown."""
        if not trader:
            return 1.0

        balance = None
        peak = getattr(trader, 'peak_balance', None)

        advisor = getattr(trader, 'advisor', None)
        if advisor:
            balance = getattr(advisor, 'current_balance', None)

        if not balance or not peak or peak <= 0:
            return 1.0

        dd_pct = (peak - balance) / peak * 100

        if dd_pct >= 5.0:
            return 0.3   # Heavy drawdown — minimal size
        elif dd_pct >= 3.0:
            return 0.5   # Moderate drawdown
        elif dd_pct >= 1.5:
            return 0.75  # Light drawdown
        else:
            return 1.0   # No significant drawdown

    def get_status(self) -> Dict:
        """Estado para dashboard."""
        return {
            "kelly_fraction": self.kelly_fraction,
            "win_rate": round(self.win_rate, 4),
            "payoff_ratio": round(self.payoff_ratio, 3),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "raw_kelly": round(self.raw_kelly, 4),
            "adjusted_kelly": round(self.adjusted_kelly, 4),
            "last_size_pct": round(self.last_size_pct, 3),
            "last_adjustments": self.last_adjustments,
            "n_trades_in_window": len(self.trade_history),
            "min_max_pct": [self.min_size_pct, self.max_size_pct],
        }
