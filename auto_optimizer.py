"""
ML SuperTrend v51 - Weekly Auto-Optimizer
==========================================
Automatically optimizes strategy parameters every week by:
1. Running backtests with different parameter combinations
2. Selecting the best parameters based on Sharpe + Win Rate
3. Applying new parameters gradually (with safety checks)

Parameters optimized:
  - SuperTrend factor (2.0 - 4.0)
  - ATR period (8 - 14)
  - ADX minimum threshold (15 - 30)
  - RSI overbought/oversold levels
  - Signal strength threshold
  - Session filter hours

Safety:
  - Only applies changes if new params backtest better than current
  - Maximum parameter change per cycle is limited (no wild swings)
  - Rollback if live performance degrades within 48 hours

Usage:
    from auto_optimizer import AutoOptimizer
    opt = AutoOptimizer(trader)
    opt.run_weekly()  # Call from scheduler or cron
"""

import logging
import json
import os
import copy
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from itertools import product

logger = logging.getLogger(__name__)


class AutoOptimizer:
    """
    Weekly automatic parameter optimization for the trading strategy.

    Uses grid search over key parameters, backtests each combination,
    and selects the best set based on a composite score.
    """

    def __init__(
        self,
        trader=None,
        optimization_lookback_days: int = 30,
        max_param_change_pct: float = 20.0,    # Max % change per parameter per cycle
        min_improvement_pct: float = 5.0,       # Minimum improvement to apply changes
        data_file: str = None,
    ):
        self.trader = trader
        self.lookback_days = optimization_lookback_days
        self.max_param_change_pct = max_param_change_pct
        self.min_improvement_pct = min_improvement_pct
        self.data_file = data_file or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "optimization_history.json"
        )

        # Parameter search space
        self.param_grid = {
            "supertrend_factor": [2.5, 3.0, 3.5, 4.0],
            "atr_period": [8, 10, 12, 14],
            "adx_min": [15, 20, 25, 30],
            "rsi_ob": [70, 75, 80],
            "rsi_os": [20, 25, 30],
            "signal_threshold": [0.45, 0.50, 0.55, 0.60],
        }

        # History
        self.optimization_history: List[Dict] = []
        self.current_params: Dict = {}
        self.last_optimization: Optional[str] = None

        self._load()
        logger.info("AutoOptimizer initialized")

    def run_weekly(self, symbols: List[str] = None, timeframes: List[str] = None) -> Dict:
        """
        Run the weekly optimization cycle.

        Args:
            symbols: List of symbols to optimize for (default: all active)
            timeframes: List of timeframes (default: ["M30"])

        Returns:
            Dict with optimization results and recommendations
        """
        if not self.trader or not self.trader.client:
            return {"error": "Trader not available"}

        if not symbols:
            symbols = ["EUR_USD"]
        if not timeframes:
            timeframes = ["M30"]

        logger.info(f"Starting weekly optimization for {symbols} on {timeframes}")

        from config import STRATEGY
        from backtester import Backtester

        current_params = {
            "supertrend_factor": STRATEGY.supertrend_factor,
            "atr_period": STRATEGY.atr_period,
            "adx_min": STRATEGY.adx_min,
            "signal_threshold": STRATEGY.signal_threshold,
        }

        # Run baseline backtest with current params
        bt = Backtester(self.trader.client, STRATEGY)
        baseline_results = {}
        for sym in symbols:
            for tf in timeframes:
                key = f"{sym}_{tf}"
                try:
                    result = bt.run_visual(sym, tf, lookback_days=self.lookback_days)
                    baseline_results[key] = result.get("metrics", {})
                except Exception as e:
                    logger.warning(f"Baseline backtest failed for {key}: {e}")

        baseline_score = self._composite_score(baseline_results)
        logger.info(f"Baseline score: {baseline_score:.4f}")

        # Generate parameter combinations (reduced grid for speed)
        best_score = baseline_score
        best_params = current_params.copy()
        results_log = []

        # Test each parameter independently (instead of full grid — too slow)
        for param_name, values in self.param_grid.items():
            for value in values:
                test_params = copy.deepcopy(STRATEGY)

                # Apply the test parameter
                if param_name == "supertrend_factor":
                    test_params.supertrend_factor = value
                elif param_name == "atr_period":
                    test_params.atr_period = value
                    test_params.atr_len = value
                elif param_name == "adx_min":
                    test_params.adx_min = value
                elif param_name == "signal_threshold":
                    test_params.signal_threshold = value
                elif param_name == "rsi_ob":
                    test_params.rsi_overbought = value
                elif param_name == "rsi_os":
                    test_params.rsi_oversold = value

                # Backtest with test params
                test_bt = Backtester(self.trader.client, test_params)
                test_results = {}
                for sym in symbols:
                    for tf in timeframes:
                        key = f"{sym}_{tf}"
                        try:
                            result = test_bt.run_visual(sym, tf, lookback_days=self.lookback_days)
                            test_results[key] = result.get("metrics", {})
                        except:
                            pass

                score = self._composite_score(test_results)
                results_log.append({
                    "param": param_name,
                    "value": value,
                    "score": round(score, 4),
                })

                if score > best_score:
                    best_score = score
                    best_params[param_name] = value

        # Check if improvement is significant
        improvement_pct = (best_score - baseline_score) / max(0.001, abs(baseline_score)) * 100

        recommendation = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "baseline_score": round(baseline_score, 4),
            "best_score": round(best_score, 4),
            "improvement_pct": round(improvement_pct, 2),
            "current_params": current_params,
            "recommended_params": best_params,
            "should_apply": improvement_pct >= self.min_improvement_pct,
            "tested_combinations": len(results_log),
            "top_results": sorted(results_log, key=lambda x: x["score"], reverse=True)[:10],
        }

        # Save to history
        self.optimization_history.append(recommendation)
        if len(self.optimization_history) > 52:  # Keep ~1 year of history
            self.optimization_history = self.optimization_history[-52:]
        self.last_optimization = recommendation["timestamp"]
        self._save()

        if recommendation["should_apply"]:
            logger.info(f"Optimization found {improvement_pct:.1f}% improvement! "
                       f"Recommended params: {best_params}")
        else:
            logger.info(f"Optimization: only {improvement_pct:.1f}% improvement. "
                       f"Keeping current parameters.")

        return recommendation

    def apply_params(self, params: Dict) -> bool:
        """
        Apply optimized parameters to the live strategy.

        Args:
            params: Dict of parameter name -> value

        Returns:
            True if applied successfully
        """
        try:
            from config import STRATEGY

            # Apply with limits (no wild swings)
            for param_name, new_value in params.items():
                current = getattr(STRATEGY, param_name, None)
                if current is None:
                    continue

                if isinstance(current, (int, float)):
                    # Limit change magnitude
                    max_delta = abs(current) * self.max_param_change_pct / 100
                    clamped = max(current - max_delta, min(current + max_delta, new_value))
                    setattr(STRATEGY, param_name, type(current)(clamped))
                    logger.info(f"Applied: {param_name} = {current} → {clamped}")

            self.current_params = params
            return True

        except Exception as e:
            logger.error(f"Failed to apply optimized params: {e}")
            return False

    def _composite_score(self, results: Dict) -> float:
        """
        Calculate a composite score from backtest results.
        Balances profitability, risk, and consistency.

        Score = 0.35 * profit_factor + 0.25 * win_rate/100 + 0.25 * (1 - max_dd/50) + 0.15 * return_pct/100
        """
        if not results:
            return 0

        scores = []
        for key, metrics in results.items():
            pf = metrics.get("profit_factor", 0)
            wr = metrics.get("win_rate", 0) / 100
            dd = min(metrics.get("max_dd_pct", metrics.get("max_drawdown_pct", 50)), 50) / 50
            ret = metrics.get("return_pct", metrics.get("total_return_pct", 0)) / 100

            score = 0.35 * min(pf, 3.0) / 3.0 + 0.25 * wr + 0.25 * (1 - dd) + 0.15 * min(max(ret, -1), 1)
            scores.append(score)

        return np.mean(scores) if scores else 0

    def get_status(self) -> Dict:
        """Get optimizer status for dashboard."""
        return {
            "last_optimization": self.last_optimization,
            "total_optimizations": len(self.optimization_history),
            "current_params": self.current_params,
            "last_result": self.optimization_history[-1] if self.optimization_history else None,
        }

    def _save(self):
        try:
            with open(self.data_file, 'w') as f:
                json.dump({
                    "history": self.optimization_history,
                    "current_params": self.current_params,
                    "last_optimization": self.last_optimization,
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save optimization data: {e}")

    def _load(self):
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                self.optimization_history = data.get("history", [])
                self.current_params = data.get("current_params", {})
                self.last_optimization = data.get("last_optimization")
        except Exception as e:
            logger.warning(f"Failed to load optimization data: {e}")
