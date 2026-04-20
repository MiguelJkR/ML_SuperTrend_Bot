"""
ML SuperTrend v51 - Ensemble Model Scorer
==========================================
Combines predictions from multiple models into a single decision using
weighted voting. Each model's weight adapts based on recent accuracy.

Models:
  1. LogisticRegression (fast, interpretable baseline)
  2. XGBoost (gradient-boosted trees, captures non-linear patterns)
  3. DQN Reinforcement Learning (adapts to market regime, risk-aware)

The ensemble uses a weighted soft vote:
  ensemble_score = w1*LR_score + w2*XGB_score + w3*RL_score

Weights are updated based on each model's rolling accuracy over
the last N predictions (exponential moving average).

Usage:
    from ensemble_scorer import EnsembleScorer
    ens = EnsembleScorer()
    decision = ens.score(lr_prob=0.72, xgb_prob=0.65, rl_action="TAKE", rl_confidence=0.8)
    # decision = {"score": 0.71, "action": "TAKE", "model_agreement": 0.67, ...}
"""

import logging
import json
import os
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# EXP3 Online Learning
try:
    from advanced_learning import EXP3OnlineLearner
    EXP3_AVAILABLE = True
except ImportError:
    EXP3_AVAILABLE = False


class EnsembleScorer:
    """
    Weighted ensemble of ML models for trade signal scoring.

    Features:
    1. Soft voting with adaptive weights
    2. Model agreement metric (how many models agree)
    3. Confidence-weighted output
    4. Automatic weight decay for underperforming models
    5. Persistence of weights across sessions
    """

    def __init__(
        self,
        initial_weights: Dict[str, float] = None,
        weight_adaptation_rate: float = 0.05,    # How fast weights adapt
        min_weight: float = 0.10,                 # Minimum weight per model
        agreement_threshold: float = 0.60,        # Require 60%+ agreement to TAKE
        high_confidence_threshold: float = 0.72,  # Score above this = strong TAKE
        low_confidence_threshold: float = 0.40,   # Score below this = BLOCK
        data_file: str = None,
    ):
        if initial_weights is None:
            initial_weights = {
                "logistic_regression": 0.30,
                "xgboost": 0.40,
                "dqn_rl": 0.30,
            }

        self.weights = initial_weights.copy()
        self.weight_adaptation_rate = weight_adaptation_rate
        self.min_weight = min_weight
        self.agreement_threshold = agreement_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.data_file = data_file or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ensemble_weights.json"
        )

        # Rolling accuracy per model (last 50 predictions)
        self.prediction_history: Dict[str, List[Dict]] = {
            "logistic_regression": [],
            "xgboost": [],
            "dqn_rl": [],
        }
        self.max_history = 50

        # Stats
        self.total_scores = 0
        self.total_takes = 0
        self.total_blocks = 0

        # ===== EXP3 Online Learning =====
        self.exp3 = None
        if EXP3_AVAILABLE:
            try:
                self.exp3 = EXP3OnlineLearner(
                    n_experts=3,
                    expert_names=["logistic_regression", "xgboost", "dqn_rl"],
                    eta=0.1,
                    gamma=0.05,
                )
                self.exp3.load()
                logger.info("EXP3 Online Learning initialized (\u03b7=0.1, \u03b3=0.05)")
            except Exception as e:
                logger.warning(f"EXP3 init error: {e}")

        # Load persisted weights
        self._load()

        logger.info(f"EnsembleScorer initialized: weights={self.weights}")

    def score(
        self,
        lr_prob: float = None,          # LogisticRegression probability (0-1)
        xgb_prob: float = None,         # XGBoost probability (0-1)
        rl_action: str = None,          # DQN action: "TAKE" or "SKIP"
        rl_confidence: float = 0.5,     # DQN confidence (0-1)
        signal_strength: float = 0.5,   # Base signal strength for fallback
    ) -> Dict:
        """
        Combine model predictions into an ensemble decision.

        Returns:
            {
                "score": float,           # 0-1 ensemble probability
                "action": str,            # "TAKE", "NEUTRAL", "BLOCK"
                "strength_multiplier": float,  # 0.7-1.2 applied to signal strength
                "model_agreement": float,  # 0-1 how much models agree
                "model_scores": dict,      # Individual model contributions
                "weights_used": dict,      # Current model weights
                "reason": str,
            }
        """
        self.total_scores += 1
        model_scores = {}
        active_weights = {}
        total_weight = 0

        # Normalize RL action to probability
        if rl_action is not None:
            rl_prob = rl_confidence if rl_action == "TAKE" else (1.0 - rl_confidence)
        else:
            rl_prob = None

        # Collect available model scores
        if lr_prob is not None:
            model_scores["logistic_regression"] = lr_prob
            active_weights["logistic_regression"] = self.weights["logistic_regression"]
            total_weight += self.weights["logistic_regression"]

        if xgb_prob is not None:
            model_scores["xgboost"] = xgb_prob
            active_weights["xgboost"] = self.weights["xgboost"]
            total_weight += self.weights["xgboost"]

        if rl_prob is not None:
            model_scores["dqn_rl"] = rl_prob
            active_weights["dqn_rl"] = self.weights["dqn_rl"]
            total_weight += self.weights["dqn_rl"]

        # Fallback: no models available
        if not model_scores:
            return {
                "score": signal_strength,
                "action": "NEUTRAL",
                "strength_multiplier": 1.0,
                "model_agreement": 0.0,
                "model_scores": {},
                "weights_used": {},
                "reason": "no_models_available",
            }

        # Normalize weights to sum to 1.0
        if total_weight > 0:
            norm_weights = {k: v / total_weight for k, v in active_weights.items()}
        else:
            norm_weights = {k: 1.0 / len(active_weights) for k in active_weights}

        # Weighted soft vote (standard)
        ensemble_score = sum(
            model_scores[m] * norm_weights[m] for m in model_scores
        )

        # EXP3 Online Learning: secondary score for comparison & blending
        exp3_score = None
        if self.exp3 and model_scores:
            try:
                exp3_score = self.exp3.get_ensemble_score(model_scores)
                # Blend: 70% standard + 30% EXP3 (EXP3 adapts faster to regime)
                ensemble_score = 0.7 * ensemble_score + 0.3 * exp3_score
            except Exception:
                pass

        # Model agreement: do models agree on direction?
        take_votes = sum(1 for s in model_scores.values() if s >= 0.5)
        total_models = len(model_scores)
        agreement = take_votes / total_models if total_models > 0 else 0

        # Decision logic
        if ensemble_score >= self.high_confidence_threshold and agreement >= self.agreement_threshold:
            action = "TAKE"
            strength_mult = 1.0 + (ensemble_score - self.high_confidence_threshold) * 0.5  # 1.0-1.14
            strength_mult = min(1.20, strength_mult)
            self.total_takes += 1
            reason = f"ensemble_high({ensemble_score:.2f},agree={agreement:.0%})"

        elif ensemble_score >= 0.50 and agreement >= self.agreement_threshold:
            action = "TAKE"
            strength_mult = 0.95 + (ensemble_score - 0.5) * 0.2  # 0.95-1.0
            self.total_takes += 1
            reason = f"ensemble_ok({ensemble_score:.2f},agree={agreement:.0%})"

        elif ensemble_score <= self.low_confidence_threshold:
            action = "BLOCK"
            strength_mult = 0.0
            self.total_blocks += 1
            reason = f"ensemble_block({ensemble_score:.2f})"

        elif agreement < self.agreement_threshold:
            action = "NEUTRAL"
            strength_mult = 0.85  # Slight penalty for disagreement
            reason = f"ensemble_disagree({ensemble_score:.2f},agree={agreement:.0%})"

        else:
            action = "NEUTRAL"
            strength_mult = 0.90
            reason = f"ensemble_neutral({ensemble_score:.2f})"

        return {
            "score": round(ensemble_score, 4),
            "action": action,
            "strength_multiplier": round(strength_mult, 3),
            "model_agreement": round(agreement, 2),
            "model_scores": {k: round(v, 4) for k, v in model_scores.items()},
            "weights_used": {k: round(v, 3) for k, v in norm_weights.items()},
            "reason": reason,
        }

    def record_outcome(self, trade_id: str, model_predictions: Dict[str, float], profitable: bool):
        """
        Record trade outcome to update model weights.

        Args:
            trade_id: Trade identifier
            model_predictions: {"logistic_regression": 0.72, "xgboost": 0.65, "dqn_rl": 0.8}
            profitable: Whether the trade was profitable
        """
        for model_name, prediction in model_predictions.items():
            if model_name not in self.prediction_history:
                continue

            # Did this model predict correctly?
            predicted_take = prediction >= 0.5
            correct = (predicted_take and profitable) or (not predicted_take and not profitable)

            self.prediction_history[model_name].append({
                "trade_id": trade_id,
                "prediction": prediction,
                "profitable": profitable,
                "correct": correct,
                "time": datetime.now(timezone.utc).isoformat(),
            })

            # Trim history
            if len(self.prediction_history[model_name]) > self.max_history:
                self.prediction_history[model_name] = self.prediction_history[model_name][-self.max_history:]

        # Update weights based on rolling accuracy
        self._update_weights()

        # EXP3 online update (fast adaptation)
        if self.exp3:
            try:
                expert_rewards = {}
                for model_name, prediction in model_predictions.items():
                    predicted_take = prediction >= 0.5
                    correct = (predicted_take and profitable) or (not predicted_take and not profitable)
                    expert_rewards[model_name] = 1.0 if correct else 0.0
                self.exp3.update(expert_rewards)
                self.exp3.save()
            except Exception as e:
                logger.warning(f"EXP3 update error: {e}")

        self._save()

    def _update_weights(self):
        """Adapt model weights based on recent accuracy."""
        accuracies = {}
        for model_name, history in self.prediction_history.items():
            if len(history) >= 5:  # Need at least 5 predictions
                correct_count = sum(1 for h in history[-20:] if h["correct"])
                total_count = len(history[-20:])
                accuracies[model_name] = correct_count / total_count
            else:
                accuracies[model_name] = 0.5  # Neutral until enough data

        if not accuracies:
            return

        # Softmax-style weight update
        total_acc = sum(accuracies.values())
        if total_acc > 0:
            for model_name, acc in accuracies.items():
                target_weight = acc / total_acc
                # Smooth adaptation
                current = self.weights.get(model_name, 0.33)
                new_weight = current + self.weight_adaptation_rate * (target_weight - current)
                self.weights[model_name] = max(self.min_weight, new_weight)

        # Renormalize
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info(f"Ensemble weights updated: {', '.join(f'{k}={v:.2%}' for k, v in self.weights.items())}")

    def get_status(self) -> Dict:
        """Get ensemble status for dashboard."""
        model_stats = {}
        for model_name, history in self.prediction_history.items():
            if history:
                recent = history[-20:]
                accuracy = sum(1 for h in recent if h["correct"]) / len(recent) if recent else 0
                model_stats[model_name] = {
                    "weight": self.weights.get(model_name, 0),
                    "accuracy_20": accuracy,
                    "total_predictions": len(history),
                }
            else:
                model_stats[model_name] = {
                    "weight": self.weights.get(model_name, 0),
                    "accuracy_20": 0,
                    "total_predictions": 0,
                }

        result = {
            "models": model_stats,
            "total_scores": self.total_scores,
            "total_takes": self.total_takes,
            "total_blocks": self.total_blocks,
            "take_rate": self.total_takes / max(1, self.total_scores),
        }

        if self.exp3:
            result["exp3"] = self.exp3.get_status()

        return result

    def _save(self):
        """Persist weights and history."""
        try:
            data = {
                "weights": self.weights,
                "prediction_history": self.prediction_history,
                "stats": {
                    "total_scores": self.total_scores,
                    "total_takes": self.total_takes,
                    "total_blocks": self.total_blocks,
                },
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save ensemble data: {e}")

    def _load(self):
        """Load persisted weights and history."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                self.weights = data.get("weights", self.weights)
                self.prediction_history = data.get("prediction_history", self.prediction_history)
                stats = data.get("stats", {})
                self.total_scores = stats.get("total_scores", 0)
                self.total_takes = stats.get("total_takes", 0)
                self.total_blocks = stats.get("total_blocks", 0)
                logger.info(f"Ensemble data loaded: {self.total_scores} historical scores")
        except Exception as e:
            logger.warning(f"Failed to load ensemble data: {e}")
