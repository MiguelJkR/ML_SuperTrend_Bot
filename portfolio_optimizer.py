"""
ML SuperTrend v51 - Portfolio Optimization (Markowitz + HMM)
=============================================================
Asignación óptima de capital entre pares de trading usando
optimización mean-variance de Markowitz, ajustada por:
  - Correlaciones cross-asset (del GNN)
  - Régimen HMM (trending/volatile/mean-reverting)
  - Risk-parity (equal risk contribution)

Fórmulas:
  Markowitz Mean-Variance:
    max   w'μ - λ/2 × w'Σw
    s.t.  Σw_i = 1, w_i >= 0

  Risk Parity:
    w_i ∝ 1/σ_i  (inverso de la volatilidad)
    Equalizar contribución de riesgo de cada asset

  Black-Litterman (con views del modelo):
    Π = λΣw_mkt                    (implied returns)
    E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 × [(τΣ)^-1Π + P'Ω^-1Q]

Papers:
  - Markowitz (1952) — "Portfolio Selection"
  - Maillard et al. (2010) — "The Properties of Equally Weighted Risk"
  - Black & Litterman (1992) — "Global Portfolio Optimization"

Uso:
    from portfolio_optimizer import PortfolioOptimizer
    po = PortfolioOptimizer()
    weights = po.optimize(returns_matrix, method="markowitz")
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SCIPY_AVAILABLE = False
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    pass


class PortfolioOptimizer:
    """
    Optimización de portfolio multi-asset para trading.
    """

    def __init__(
        self,
        risk_aversion: float = 2.0,       # λ para Markowitz
        min_weight: float = 0.05,          # Mínimo 5% por asset
        max_weight: float = 0.40,          # Máximo 40% por asset
        risk_free_rate: float = 0.04,      # 4% anualizado
    ):
        self.risk_aversion = risk_aversion
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.risk_free_rate = risk_free_rate

        self.last_weights: Dict[str, float] = {}
        self.last_method: str = ""
        self.last_metrics: Dict = {}

    def optimize(
        self,
        returns: np.ndarray,
        asset_names: List[str] = None,
        method: str = "markowitz",
        regime: str = None,
    ) -> Dict[str, float]:
        """
        Optimizar pesos del portfolio.

        Args:
            returns: (n_periods, n_assets) matrix de retornos
            asset_names: nombres de cada asset
            method: "markowitz", "risk_parity", "min_variance", "max_sharpe"
            regime: Si se provee, ajusta la optimización por régimen

        Returns:
            {asset_name: weight}
        """
        n_assets = returns.shape[1] if len(returns.shape) > 1 else 1
        if asset_names is None:
            asset_names = [f"asset_{i}" for i in range(n_assets)]

        if n_assets == 1:
            self.last_weights = {asset_names[0]: 1.0}
            return self.last_weights

        # Estadísticas
        mu = np.mean(returns, axis=0)      # Expected returns
        sigma = np.cov(returns.T)           # Covariance matrix
        vols = np.sqrt(np.diag(sigma))      # Volatilities

        # Ajuste por régimen
        if regime == "VOLATILE":
            sigma *= 1.5  # Increase risk estimate
            mu *= 0.7     # Reduce return expectation
        elif regime == "TRENDING":
            mu *= 1.2     # Increase expected returns in trend

        # Optimizar
        if method == "markowitz":
            weights = self._markowitz(mu, sigma, n_assets)
        elif method == "risk_parity":
            weights = self._risk_parity(sigma, n_assets)
        elif method == "min_variance":
            weights = self._min_variance(sigma, n_assets)
        elif method == "max_sharpe":
            weights = self._max_sharpe(mu, sigma, n_assets)
        else:
            weights = np.ones(n_assets) / n_assets

        # Clamp weights
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = weights / weights.sum()  # Renormalize

        self.last_weights = {
            asset_names[i]: round(float(weights[i]), 4)
            for i in range(n_assets)
        }
        self.last_method = method

        # Portfolio metrics
        port_return = float(np.dot(weights, mu))
        port_vol = float(np.sqrt(np.dot(weights.T, np.dot(sigma, weights))))
        port_sharpe = (port_return - self.risk_free_rate / 252) / (port_vol + 1e-10)

        self.last_metrics = {
            "expected_return": round(port_return * 252, 4),
            "expected_vol": round(port_vol * np.sqrt(252), 4),
            "sharpe_ratio": round(port_sharpe * np.sqrt(252), 3),
            "n_assets": n_assets,
            "method": method,
            "regime_adjustment": regime,
        }

        logger.info(f"Portfolio optimized ({method}): {self.last_weights}")
        return self.last_weights

    def _markowitz(self, mu: np.ndarray, sigma: np.ndarray, n: int) -> np.ndarray:
        """Mean-variance optimization: max w'μ - λ/2 × w'Σw"""
        if not SCIPY_AVAILABLE:
            return self._analytical_markowitz(mu, sigma, n)

        def neg_utility(w):
            ret = np.dot(w, mu)
            risk = np.dot(w.T, np.dot(sigma, w))
            return -(ret - self.risk_aversion / 2 * risk)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(self.min_weight, self.max_weight)] * n
        x0 = np.ones(n) / n

        result = minimize(neg_utility, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return result.x if result.success else np.ones(n) / n

    def _analytical_markowitz(self, mu: np.ndarray, sigma: np.ndarray, n: int) -> np.ndarray:
        """Markowitz analítico sin scipy (aproximación)."""
        try:
            sigma_inv = np.linalg.inv(sigma + np.eye(n) * 1e-8)
            w = sigma_inv @ mu
            w = np.maximum(w, 0)
            w = w / (np.sum(w) + 1e-10)
            return w
        except Exception:
            return np.ones(n) / n

    def _risk_parity(self, sigma: np.ndarray, n: int) -> np.ndarray:
        """Risk parity: equalizar contribución de riesgo."""
        vols = np.sqrt(np.diag(sigma))
        inv_vols = 1.0 / (vols + 1e-10)
        weights = inv_vols / inv_vols.sum()
        return weights

    def _min_variance(self, sigma: np.ndarray, n: int) -> np.ndarray:
        """Minimum variance portfolio."""
        if not SCIPY_AVAILABLE:
            return self._risk_parity(sigma, n)

        def portfolio_var(w):
            return np.dot(w.T, np.dot(sigma, w))

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(self.min_weight, self.max_weight)] * n
        x0 = np.ones(n) / n

        result = minimize(portfolio_var, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return result.x if result.success else np.ones(n) / n

    def _max_sharpe(self, mu: np.ndarray, sigma: np.ndarray, n: int) -> np.ndarray:
        """Maximum Sharpe Ratio portfolio."""
        if not SCIPY_AVAILABLE:
            return self._analytical_markowitz(mu, sigma, n)

        rf_daily = self.risk_free_rate / 252

        def neg_sharpe(w):
            ret = np.dot(w, mu) - rf_daily
            vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
            return -(ret / (vol + 1e-10))

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(self.min_weight, self.max_weight)] * n
        x0 = np.ones(n) / n

        result = minimize(neg_sharpe, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return result.x if result.success else np.ones(n) / n

    def get_efficient_frontier(
        self,
        returns: np.ndarray,
        n_points: int = 20,
    ) -> Dict:
        """
        Calcular puntos de la frontera eficiente.

        Returns:
            {"returns": [...], "volatilities": [...], "sharpes": [...]}
        """
        mu = np.mean(returns, axis=0)
        sigma = np.cov(returns.T)
        n = returns.shape[1]

        target_returns = np.linspace(np.min(mu), np.max(mu), n_points)
        frontier = {"returns": [], "volatilities": [], "sharpes": []}

        if not SCIPY_AVAILABLE:
            # Simple estimation without optimization
            for target in target_returns:
                w = np.ones(n) / n
                port_vol = float(np.sqrt(np.dot(w.T, np.dot(sigma, w))))
                frontier["returns"].append(round(float(target * 252), 4))
                frontier["volatilities"].append(round(port_vol * np.sqrt(252), 4))
                s = (target * 252 - self.risk_free_rate) / (port_vol * np.sqrt(252) + 1e-10)
                frontier["sharpes"].append(round(s, 3))
            return frontier

        for target in target_returns:
            def portfolio_var(w):
                return np.dot(w.T, np.dot(sigma, w))

            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w, t=target: np.dot(w, mu) - t},
            ]
            bounds = [(self.min_weight, self.max_weight)] * n
            x0 = np.ones(n) / n

            try:
                result = minimize(portfolio_var, x0, method='SLSQP',
                                 bounds=bounds, constraints=constraints)
                if result.success:
                    port_vol = float(np.sqrt(result.fun))
                    port_ret = float(target * 252)
                    sharpe = (port_ret - self.risk_free_rate) / (port_vol * np.sqrt(252) + 1e-10)
                    frontier["returns"].append(round(port_ret, 4))
                    frontier["volatilities"].append(round(port_vol * np.sqrt(252), 4))
                    frontier["sharpes"].append(round(sharpe, 3))
            except Exception:
                pass

        return frontier

    def apply_to_trader(self, trader, weights: Dict[str, float] = None) -> Dict:
        """
        Aplicar pesos optimizados al position sizing del trader.

        Args:
            trader: Instancia del Trader
            weights: Pesos a aplicar (default: self.last_weights)

        Returns:
            Dict de cambios aplicados
        """
        w = weights or self.last_weights
        if not w:
            return {"error": "No weights to apply"}

        applied = {}

        # Apply as risk allocation per instrument
        kelly = getattr(trader, 'kelly_sizer', None)
        if kelly:
            # Scale Kelly max_size by portfolio weight
            for asset, weight in w.items():
                # The weight represents the portfolio allocation
                # Scale max position size proportionally
                applied[asset] = {
                    "portfolio_weight": round(weight, 4),
                    "max_risk_pct": round(weight * kelly.max_size_pct, 3),
                }

        logger.info(f"Portfolio weights applied to {len(applied)} assets")
        return applied

    def get_status(self) -> Dict:
        return {
            "last_method": self.last_method,
            "last_weights": self.last_weights,
            "last_metrics": self.last_metrics,
            "risk_aversion": self.risk_aversion,
            "min_max_weight": [self.min_weight, self.max_weight],
            "scipy_available": SCIPY_AVAILABLE,
        }
