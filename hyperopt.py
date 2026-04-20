"""
ML SuperTrend v51 - Hyperparameter Optimization (Optuna/Bayesian)
===================================================================
Optimización automática de hiperparámetros usando Bayesian optimization.
Busca los mejores parámetros para LSTM, ensemble, y las 20 estrategias.

Espacios de búsqueda:
  - LSTM: learning_rate, hidden_size, num_layers, dropout, seq_length
  - Ensemble: pesos de cada modelo, umbrales de confianza
  - Risk: risk_per_trade, max_daily_loss, drawdown_limit
  - Estrategias: thresholds de Wasserstein, Fisher, MC Dropout, etc.

Integración:
  - Se ejecuta automáticamente en el weekly review (viernes)
  - Usa los últimos N trades como función objetivo
  - Función objetivo: Sharpe Ratio (o Sortino, profit_factor)

Uso:
    from hyperopt import HyperOptimizer
    opt = HyperOptimizer(trader)
    best_params = opt.optimize(n_trials=50)
    opt.apply_best_params(trader)
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Callable
import json
import os

logger = logging.getLogger(__name__)

# Try Optuna first, fallback to random search
OPTUNA_AVAILABLE = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
    logger.info("Optuna disponible — usando TPE Bayesian optimization")
except ImportError:
    logger.info("Optuna no disponible — usando random search fallback")


class SearchSpace:
    """Definición del espacio de búsqueda de hiperparámetros."""

    LSTM_PARAMS = {
        "lstm_lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        "lstm_hidden": {"type": "categorical", "choices": [32, 64, 128, 256]},
        "lstm_layers": {"type": "int", "low": 1, "high": 3},
        "lstm_dropout": {"type": "uniform", "low": 0.1, "high": 0.5},
        "lstm_seq_length": {"type": "categorical", "choices": [15, 20, 30, 50]},
        "lstm_batch_size": {"type": "categorical", "choices": [32, 64, 128]},
    }

    ENSEMBLE_PARAMS = {
        "weight_lr": {"type": "uniform", "low": 0.1, "high": 0.5},
        "weight_xgb": {"type": "uniform", "low": 0.1, "high": 0.5},
        "weight_rl": {"type": "uniform", "low": 0.1, "high": 0.5},
        "confidence_threshold": {"type": "uniform", "low": 0.50, "high": 0.75},
        "high_conf_threshold": {"type": "uniform", "low": 0.65, "high": 0.85},
    }

    RISK_PARAMS = {
        "risk_per_trade": {"type": "uniform", "low": 0.5, "high": 2.0},
        "max_daily_loss_pct": {"type": "uniform", "low": 2.0, "high": 5.0},
        "sl_multiplier": {"type": "uniform", "low": 1.0, "high": 3.0},
        "tp_multiplier": {"type": "uniform", "low": 1.5, "high": 4.0},
    }

    STRATEGY_PARAMS = {
        "wasserstein_alert": {"type": "uniform", "low": 0.3, "high": 1.0},
        "wasserstein_critical": {"type": "uniform", "low": 0.8, "high": 2.0},
        "fisher_alert": {"type": "uniform", "low": 1.5, "high": 3.0},
        "fisher_critical": {"type": "uniform", "low": 2.5, "high": 5.0},
        "mc_dropout_threshold": {"type": "uniform", "low": 0.10, "high": 0.25},
        "mc_dropout_passes": {"type": "categorical", "choices": [10, 15, 20, 30]},
        "curriculum_initial_pct": {"type": "uniform", "low": 0.2, "high": 0.5},
        "curriculum_growth": {"type": "uniform", "low": 0.05, "high": 0.15},
        "sharpe_alpha": {"type": "uniform", "low": 0.5, "high": 0.9},
        "exp3_eta": {"type": "loguniform", "low": 0.01, "high": 0.5},
        "exp3_gamma": {"type": "uniform", "low": 0.01, "high": 0.1},
    }


class HyperOptimizer:
    """
    Optimizador de hiperparámetros con Optuna (TPE) o random search.
    """

    def __init__(
        self,
        objective_metric: str = "sharpe",   # sharpe, sortino, profit_factor, net_pnl
        n_trials: int = 50,
        search_spaces: List[str] = None,    # ["lstm", "ensemble", "risk", "strategy"]
    ):
        self.objective_metric = objective_metric
        self.n_trials = n_trials
        self.search_spaces = search_spaces or ["lstm", "ensemble", "risk", "strategy"]

        self.best_params: Dict = {}
        self.best_score: float = -np.inf
        self.optimization_history: List[Dict] = []
        self.study = None

    def optimize(
        self,
        evaluate_func: Callable[[Dict], float],
        n_trials: int = None,
    ) -> Dict:
        """
        Ejecutar optimización.

        Args:
            evaluate_func: Función que recibe params dict y retorna score
                           (mayor es mejor, e.g., Sharpe ratio)
            n_trials: Override del número de trials

        Returns:
            best_params dict
        """
        n = n_trials or self.n_trials

        if OPTUNA_AVAILABLE:
            return self._optimize_optuna(evaluate_func, n)
        else:
            return self._optimize_random(evaluate_func, n)

    def _optimize_optuna(self, evaluate_func: Callable, n_trials: int) -> Dict:
        """Optimización con Optuna TPE sampler."""
        def objective(trial):
            params = self._sample_optuna(trial)
            try:
                score = evaluate_func(params)
                return score
            except Exception as e:
                logger.debug(f"Trial failed: {e}")
                return -999

        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )

        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        self.optimization_history = [
            {
                "trial": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in self.study.trials
        ]

        logger.info(f"Optuna optimization complete: {n_trials} trials, "
                   f"best {self.objective_metric}={self.best_score:.4f}")

        return self.best_params

    def _sample_optuna(self, trial) -> Dict:
        """Sample parameters using Optuna trial."""
        params = {}
        all_spaces = self._get_all_spaces()

        for name, spec in all_spaces.items():
            if spec["type"] == "uniform":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
            elif spec["type"] == "loguniform":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])

        # Normalize ensemble weights
        if all(k in params for k in ["weight_lr", "weight_xgb", "weight_rl"]):
            total = params["weight_lr"] + params["weight_xgb"] + params["weight_rl"]
            params["weight_lr"] /= total
            params["weight_xgb"] /= total
            params["weight_rl"] /= total

        return params

    def _optimize_random(self, evaluate_func: Callable, n_trials: int) -> Dict:
        """Random search fallback cuando Optuna no está disponible."""
        logger.info(f"Starting random search: {n_trials} trials")

        all_spaces = self._get_all_spaces()

        for trial_n in range(n_trials):
            params = {}
            for name, spec in all_spaces.items():
                if spec["type"] == "uniform":
                    params[name] = np.random.uniform(spec["low"], spec["high"])
                elif spec["type"] == "loguniform":
                    log_val = np.random.uniform(np.log(spec["low"]), np.log(spec["high"]))
                    params[name] = np.exp(log_val)
                elif spec["type"] == "int":
                    params[name] = np.random.randint(spec["low"], spec["high"] + 1)
                elif spec["type"] == "categorical":
                    params[name] = np.random.choice(spec["choices"])

            # Normalize ensemble weights
            if all(k in params for k in ["weight_lr", "weight_xgb", "weight_rl"]):
                total = params["weight_lr"] + params["weight_xgb"] + params["weight_rl"]
                params["weight_lr"] /= total
                params["weight_xgb"] /= total
                params["weight_rl"] /= total

            try:
                score = evaluate_func(params)
            except Exception:
                score = -999

            self.optimization_history.append({
                "trial": trial_n, "value": score, "params": params
            })

            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                logger.info(f"  Trial {trial_n}: NEW BEST {self.objective_metric}={score:.4f}")

        logger.info(f"Random search complete: best {self.objective_metric}={self.best_score:.4f}")
        return self.best_params

    def _get_all_spaces(self) -> Dict:
        """Get combined search space based on selected spaces."""
        all_spaces = {}
        if "lstm" in self.search_spaces:
            all_spaces.update(SearchSpace.LSTM_PARAMS)
        if "ensemble" in self.search_spaces:
            all_spaces.update(SearchSpace.ENSEMBLE_PARAMS)
        if "risk" in self.search_spaces:
            all_spaces.update(SearchSpace.RISK_PARAMS)
        if "strategy" in self.search_spaces:
            all_spaces.update(SearchSpace.STRATEGY_PARAMS)
        return all_spaces

    def apply_best_params(self, trader) -> Dict:
        """
        Aplicar los mejores parámetros encontrados al trader.
        Solo aplica parámetros que se pueden cambiar en runtime.

        Returns: dict de cambios aplicados
        """
        if not self.best_params:
            return {"error": "No optimization results"}

        applied = {}

        # Ensemble weights
        ens = getattr(trader, 'ensemble_scorer', None)
        if ens:
            for key in ["weight_lr", "weight_xgb", "weight_rl"]:
                if key in self.best_params:
                    model_name = key.replace("weight_", "")
                    map_names = {"lr": "logistic_regression", "xgb": "xgboost", "rl": "dqn_rl"}
                    full_name = map_names.get(model_name, model_name)
                    if full_name in ens.weights:
                        ens.weights[full_name] = round(self.best_params[key], 3)
                        applied[f"ensemble.{full_name}"] = round(self.best_params[key], 3)

            if "confidence_threshold" in self.best_params:
                ens.agreement_threshold = round(self.best_params["confidence_threshold"], 3)
                applied["ensemble.confidence_threshold"] = round(self.best_params["confidence_threshold"], 3)

        # MC Dropout
        lstm = getattr(trader, 'lstm_predictor', None)
        if lstm and hasattr(lstm, 'mc_dropout') and lstm.mc_dropout:
            if "mc_dropout_threshold" in self.best_params:
                lstm.mc_dropout.uncertainty_threshold = self.best_params["mc_dropout_threshold"]
                applied["mc_dropout.threshold"] = round(self.best_params["mc_dropout_threshold"], 4)

        # Wasserstein thresholds
        wd = getattr(trader, 'wasserstein_drift', None)
        if wd:
            if "wasserstein_alert" in self.best_params:
                wd.alert_threshold = self.best_params["wasserstein_alert"]
                applied["wasserstein.alert"] = round(self.best_params["wasserstein_alert"], 3)
            if "wasserstein_critical" in self.best_params:
                wd.critical_threshold = self.best_params["wasserstein_critical"]
                applied["wasserstein.critical"] = round(self.best_params["wasserstein_critical"], 3)

        # Fisher thresholds
        fisher = getattr(trader, 'fisher_detector', None)
        if fisher:
            if "fisher_alert" in self.best_params:
                fisher.alert_threshold = self.best_params["fisher_alert"]
                applied["fisher.alert"] = round(self.best_params["fisher_alert"], 3)
            if "fisher_critical" in self.best_params:
                fisher.critical_threshold = self.best_params["fisher_critical"]
                applied["fisher.critical"] = round(self.best_params["fisher_critical"], 3)

        # EXP3
        if ens and hasattr(ens, 'exp3') and ens.exp3:
            if "exp3_eta" in self.best_params:
                ens.exp3.eta = self.best_params["exp3_eta"]
                applied["exp3.eta"] = round(self.best_params["exp3_eta"], 4)

        logger.info(f"Applied {len(applied)} optimized parameters: {applied}")
        return applied

    def create_evaluate_func(self, trades: List[Dict]) -> Callable:
        """
        Crear función de evaluación basada en trades históricos.
        Simula qué pasaría con diferentes parámetros.
        """
        def evaluate(params: Dict) -> float:
            if not trades:
                return -999

            # Filter trades by confidence threshold
            conf_threshold = params.get("confidence_threshold", 0.55)
            filtered = [t for t in trades
                       if t.get("confidence", 1.0) >= conf_threshold]

            if len(filtered) < 5:
                return -999

            pnls = np.array([t.get("pnl", 0) for t in filtered])

            # Apply risk sizing
            risk = params.get("risk_per_trade", 1.0)
            pnls_adjusted = pnls * (risk / 1.0)  # Scale relative to base 1%

            # Calculate objective metric
            if self.objective_metric == "sharpe":
                mean_r = np.mean(pnls_adjusted)
                std_r = np.std(pnls_adjusted) + 1e-10
                return float(mean_r / std_r * np.sqrt(252))

            elif self.objective_metric == "sortino":
                mean_r = np.mean(pnls_adjusted)
                downside = pnls_adjusted[pnls_adjusted < 0]
                down_std = np.std(downside) + 1e-10 if len(downside) > 0 else 1e-10
                return float(mean_r / down_std * np.sqrt(252))

            elif self.objective_metric == "profit_factor":
                gains = pnls_adjusted[pnls_adjusted > 0]
                losses = np.abs(pnls_adjusted[pnls_adjusted < 0])
                return float(np.sum(gains) / (np.sum(losses) + 0.01))

            elif self.objective_metric == "net_pnl":
                return float(np.sum(pnls_adjusted))

            return float(np.mean(pnls_adjusted))

        return evaluate

    def save_results(self, filepath: str = None):
        """Guardar resultados de optimización."""
        if filepath is None:
            filepath = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "training_logs",
                f"hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "objective": self.objective_metric,
            "n_trials": len(self.optimization_history),
            "best_score": round(self.best_score, 6),
            "best_params": {k: round(v, 6) if isinstance(v, float) else v
                           for k, v in self.best_params.items()},
            "optuna_available": OPTUNA_AVAILABLE,
            "history": self.optimization_history[-20:],  # Last 20 trials
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"HyperOpt results saved: {filepath}")

    def get_status(self) -> Dict:
        return {
            "optuna_available": OPTUNA_AVAILABLE,
            "objective": self.objective_metric,
            "n_trials_completed": len(self.optimization_history),
            "best_score": round(self.best_score, 4) if self.best_score > -np.inf else None,
            "best_params": {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in self.best_params.items()},
            "search_spaces": self.search_spaces,
        }
