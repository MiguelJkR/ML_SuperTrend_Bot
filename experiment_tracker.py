"""
ML SuperTrend v51 - Experiment Tracker
========================================
Registro automático de cada entrenamiento con parámetros, métricas y
modelo. Permite comparar versiones y rollback a la mejor configuración.

Funcionalidades:
  - Log de cada entrenamiento: params, metrics, timestamp, model hash
  - Comparación side-by-side de experimentos
  - Auto-detect best experiment by metric
  - Rollback: restaurar params de un experimento anterior
  - Leaderboard: top N configuraciones por métrica

Estructura de almacenamiento:
  experiments/
    experiment_001.json
    experiment_002.json
    ...
    leaderboard.json

Uso:
    from experiment_tracker import ExperimentTracker
    tracker = ExperimentTracker()
    exp_id = tracker.log_experiment(params, metrics, model_hash)
    best = tracker.get_best_experiment("sharpe_ratio")
    tracker.rollback(trader, exp_id)
"""

import logging
import json
import os
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Registra y compara entrenamientos para encontrar
    la mejor configuración del bot.
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "experiments"
        )
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        self.experiments: List[Dict] = []
        self.next_id: int = 1
        self._load_experiments()

    def log_experiment(
        self,
        params: Dict,
        metrics: Dict,
        model_hash: str = None,
        tags: List[str] = None,
        notes: str = "",
    ) -> str:
        """
        Registrar un nuevo experimento.

        Args:
            params: Hiperparámetros usados
            metrics: Métricas resultantes (sharpe, pf, win_rate, etc.)
            model_hash: Hash del modelo entrenado (para verificar)
            tags: Etiquetas opcionales ["nightly", "weekly", "manual"]
            notes: Notas del experimentador

        Returns:
            experiment_id
        """
        exp_id = f"exp_{self.next_id:04d}"
        self.next_id += 1

        experiment = {
            "id": exp_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params": params,
            "metrics": metrics,
            "model_hash": model_hash or self._hash_params(params),
            "tags": tags or [],
            "notes": notes,
        }

        self.experiments.append(experiment)
        self._save_experiment(experiment)
        self._update_leaderboard()

        logger.info(f"Experiment logged: {exp_id} \u2014 "
                   f"sharpe={metrics.get('sharpe_ratio', 'N/A')}, "
                   f"pf={metrics.get('profit_factor', 'N/A')}")

        return exp_id

    def log_training_run(self, trader) -> str:
        """
        Capturar automáticamente el estado actual del trader como experimento.
        """
        params = self._extract_current_params(trader)
        metrics = self._extract_current_metrics(trader)

        return self.log_experiment(
            params=params,
            metrics=metrics,
            tags=["auto"],
            notes="Auto-captured from training run",
        )

    def get_best_experiment(self, metric: str = "sharpe_ratio", top_n: int = 1) -> List[Dict]:
        """Obtener los mejores N experimentos por métrica."""
        valid = [e for e in self.experiments
                if metric in e.get("metrics", {})]

        if not valid:
            return []

        sorted_exps = sorted(
            valid,
            key=lambda e: e["metrics"].get(metric, -999),
            reverse=True
        )

        return sorted_exps[:top_n]

    def compare(self, exp_id_1: str, exp_id_2: str) -> Dict:
        """Comparar dos experimentos side-by-side."""
        e1 = self._find_experiment(exp_id_1)
        e2 = self._find_experiment(exp_id_2)

        if not e1 or not e2:
            return {"error": "Experiment not found"}

        comparison = {
            "exp_1": exp_id_1,
            "exp_2": exp_id_2,
            "metrics_diff": {},
            "params_diff": {},
        }

        # Compare metrics
        all_metrics = set(list(e1.get("metrics", {}).keys()) +
                         list(e2.get("metrics", {}).keys()))
        for m in all_metrics:
            v1 = e1.get("metrics", {}).get(m, None)
            v2 = e2.get("metrics", {}).get(m, None)
            if v1 is not None and v2 is not None:
                diff = v2 - v1
                comparison["metrics_diff"][m] = {
                    "exp_1": round(v1, 4) if isinstance(v1, float) else v1,
                    "exp_2": round(v2, 4) if isinstance(v2, float) else v2,
                    "diff": round(diff, 4) if isinstance(diff, float) else diff,
                    "better": "exp_2" if diff > 0 else "exp_1" if diff < 0 else "equal",
                }

        # Compare params
        all_params = set(list(e1.get("params", {}).keys()) +
                        list(e2.get("params", {}).keys()))
        for p in all_params:
            v1 = e1.get("params", {}).get(p, None)
            v2 = e2.get("params", {}).get(p, None)
            if v1 != v2:
                comparison["params_diff"][p] = {"exp_1": v1, "exp_2": v2}

        return comparison

    def rollback(self, trader, exp_id: str) -> Dict:
        """
        Restaurar parámetros de un experimento anterior al trader.

        Returns: dict de cambios aplicados
        """
        exp = self._find_experiment(exp_id)
        if not exp:
            return {"error": f"Experiment {exp_id} not found"}

        params = exp.get("params", {})
        applied = {}

        # Apply ensemble weights
        ens = getattr(trader, 'ensemble_scorer', None)
        if ens and "ensemble_weights" in params:
            for model_name, weight in params["ensemble_weights"].items():
                if model_name in ens.weights:
                    ens.weights[model_name] = weight
                    applied[f"ensemble.{model_name}"] = weight

        # Apply confidence thresholds
        if ens and "confidence_threshold" in params:
            ens.agreement_threshold = params["confidence_threshold"]
            applied["confidence_threshold"] = params["confidence_threshold"]

        # Apply MC Dropout threshold
        lstm = getattr(trader, 'lstm_predictor', None)
        if lstm and hasattr(lstm, 'mc_dropout') and lstm.mc_dropout:
            if "mc_dropout_threshold" in params:
                lstm.mc_dropout.uncertainty_threshold = params["mc_dropout_threshold"]
                applied["mc_dropout_threshold"] = params["mc_dropout_threshold"]

        # Apply Kelly fraction
        kelly = getattr(trader, 'kelly_sizer', None)
        if kelly and "kelly_fraction" in params:
            kelly.kelly_fraction = params["kelly_fraction"]
            applied["kelly_fraction"] = params["kelly_fraction"]

        logger.info(f"Rollback to {exp_id}: applied {len(applied)} params")
        return {"experiment": exp_id, "applied": applied}

    def get_leaderboard(self, metric: str = "sharpe_ratio", top_n: int = 10) -> List[Dict]:
        """Top N experimentos por métrica."""
        return self.get_best_experiment(metric, top_n)

    def _extract_current_params(self, trader) -> Dict:
        """Extraer parámetros actuales del trader."""
        params = {}

        # Ensemble weights
        ens = getattr(trader, 'ensemble_scorer', None)
        if ens and hasattr(ens, 'weights'):
            params["ensemble_weights"] = dict(ens.weights)
            params["confidence_threshold"] = getattr(ens, 'agreement_threshold', 0.6)

        # LSTM config
        lstm = getattr(trader, 'lstm_predictor', None)
        if lstm:
            params["lstm_hidden"] = getattr(lstm, 'hidden_size', 128)
            params["lstm_seq_length"] = getattr(lstm, 'sequence_length', 30)
            if hasattr(lstm, 'mc_dropout') and lstm.mc_dropout:
                params["mc_dropout_threshold"] = lstm.mc_dropout.uncertainty_threshold

        # Kelly
        kelly = getattr(trader, 'kelly_sizer', None)
        if kelly:
            params["kelly_fraction"] = kelly.kelly_fraction

        # Wasserstein thresholds
        wd = getattr(trader, 'wasserstein_drift', None)
        if wd:
            params["wasserstein_alert"] = wd.alert_threshold
            params["wasserstein_critical"] = wd.critical_threshold

        return params

    def _extract_current_metrics(self, trader) -> Dict:
        """Extraer métricas actuales."""
        metrics = {}

        lstm = getattr(trader, 'lstm_predictor', None)
        if lstm:
            total = getattr(lstm, 'total_predictions', 0)
            correct = getattr(lstm, 'correct_predictions', 0)
            metrics["lstm_accuracy"] = round(correct / max(total, 1), 4)
            metrics["lstm_predictions"] = total

        advisor = getattr(trader, 'advisor', None)
        if advisor:
            trades = getattr(advisor, 'trades_history', [])
            if trades:
                pnls = [t.get('pnl', 0) for t in trades[-50:]]
                if pnls:
                    wins = [p for p in pnls if p > 0]
                    losses = [p for p in pnls if p < 0]
                    metrics["win_rate"] = round(len(wins) / len(pnls), 4)
                    metrics["net_pnl_50"] = round(sum(pnls), 2)
                    metrics["profit_factor"] = round(
                        sum(wins) / max(abs(sum(losses)), 0.01), 2
                    ) if losses else 0
                    mean_r = np.mean(pnls) if pnls else 0
                    std_r = np.std(pnls) if len(pnls) > 1 else 1
                    metrics["sharpe_ratio"] = round(
                        float(mean_r / (std_r + 1e-10) * np.sqrt(252)), 3
                    )

        stats = getattr(trader, 'stats', {})
        metrics["total_trades"] = stats.get("total_trades", 0)

        return metrics

    def _find_experiment(self, exp_id: str) -> Optional[Dict]:
        for e in self.experiments:
            if e.get("id") == exp_id:
                return e
        return None

    def _hash_params(self, params: Dict) -> str:
        param_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(param_str.encode()).hexdigest()[:12]

    def _save_experiment(self, exp: Dict):
        filepath = os.path.join(self.data_dir, f"{exp['id']}.json")
        with open(filepath, 'w') as f:
            json.dump(exp, f, indent=2, default=str)

    def _load_experiments(self):
        """Cargar experimentos existentes."""
        import glob
        files = sorted(glob.glob(os.path.join(self.data_dir, "exp_*.json")))
        for filepath in files:
            try:
                with open(filepath) as f:
                    exp = json.load(f)
                    self.experiments.append(exp)
                    # Update next_id
                    num = int(exp["id"].split("_")[1])
                    self.next_id = max(self.next_id, num + 1)
            except Exception:
                pass

        if self.experiments:
            logger.info(f"Loaded {len(self.experiments)} experiments")

    def _update_leaderboard(self):
        """Actualizar leaderboard."""
        lb_path = os.path.join(self.data_dir, "leaderboard.json")
        leaderboard = {}

        for metric in ["sharpe_ratio", "profit_factor", "win_rate", "net_pnl_50"]:
            best = self.get_best_experiment(metric, top_n=5)
            leaderboard[metric] = [
                {
                    "id": e["id"],
                    "value": round(e["metrics"].get(metric, 0), 4),
                    "timestamp": e.get("timestamp", ""),
                }
                for e in best
            ]

        with open(lb_path, 'w') as f:
            json.dump(leaderboard, f, indent=2)

    def get_status(self) -> Dict:
        return {
            "total_experiments": len(self.experiments),
            "data_dir": self.data_dir,
            "latest": self.experiments[-1] if self.experiments else None,
            "best_sharpe": self.get_best_experiment("sharpe_ratio", 1),
        }


# Need numpy for metrics calculation
import numpy as np
