"""
ML SuperTrend v51 - Professional Live Dashboard
================================================
Real-time monitoring, trade management, regime detection, ML insights, and advisor reports.
ALL API responses are aligned with dashboard_live.html field expectations.
"""

import logging
import json
import os
from datetime import datetime, timezone
from flask import Flask, jsonify, request, Response, send_file
from config import DASHBOARD_HOST, DASHBOARD_PORT

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global references (set by main.py)
_trader = None
_regime_detector = None
_ml_scorer = None
_financial_advisor = None
_smart_risk = None
_walk_forward = None
_rl_scorer = None
_stream_manager = None


def set_modules(trader=None, regime=None, ml=None, advisor=None, risk=None, wfo=None, 
                rl_scorer=None, stream_manager=None):
    """Initialize module references from main.py"""
    global _trader, _regime_detector, _ml_scorer, _financial_advisor, _smart_risk, _walk_forward
    global _rl_scorer, _stream_manager
    _trader = trader
    _regime_detector = regime
    _ml_scorer = ml
    _financial_advisor = advisor
    _smart_risk = risk
    _walk_forward = wfo
    _rl_scorer = rl_scorer
    _stream_manager = stream_manager


# ============================================================================
# SERVE DASHBOARD
# ============================================================================

@app.route("/")
def index():
    """Serve the main dashboard HTML"""
    html_path = os.path.join(os.path.dirname(__file__), "dashboard_pro.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return Response(f.read(), mimetype="text/html")
    except FileNotFoundError:
        return jsonify({"error": "Dashboard HTML not found"}), 404


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route("/api/status")
def api_status():
    """Account status: balance, nav, unrealized_pnl, margin."""
    try:
        if not _trader or not _trader.client:
            return jsonify({
                "account": None,
                "bot_status": "offline",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 200

        try:
            account_data = _trader.client.get_account_summary()
            account_info = account_data.get("account", {}) if account_data else {}

            balance = float(account_info.get("balance", 0))
            nav = float(account_info.get("NAV", 0))
            unrealized = float(account_info.get("unrealizedPL", 0))
            margin_used_raw = float(account_info.get("marginUsed", 0))
            margin_avail_raw = float(account_info.get("marginAvailable", 0))

            total_margin = margin_used_raw + margin_avail_raw
            margin_used_pct = (margin_used_raw / total_margin * 100) if total_margin > 0 else 0
            margin_avail_pct = (margin_avail_raw / total_margin * 100) if total_margin > 0 else 100

            return jsonify({
                "account": {
                    "balance": balance,
                    "nav": nav,
                    "unrealized_pnl": unrealized,
                    "todays_pnl": float(account_info.get("pl", 0)),
                    "margin_used": margin_used_pct,
                    "margin_available": margin_avail_pct,
                    "open_trades": int(account_info.get("openTradeCount", 0)),
                    "currency": account_info.get("currency", "USD")
                },
                "bot_status": "running" if _trader._running else "stopped",
                "peak_balance": float(_trader.peak_balance) if _trader.peak_balance else 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 200
        except Exception as e:
            logger.warning(f"Error getting account summary: {e}")
            return jsonify({
                "account": None,
                "bot_status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 200

    except Exception as e:
        logger.exception("Error in /api/status")
        return jsonify({"error": str(e), "bot_status": "error"}), 500


@app.route("/api/regime")
def api_regime():
    """Market regime detection per instrument."""
    try:
        if not _regime_detector or not _trader or not _trader.client:
            return jsonify({"regimes": [], "timestamp": datetime.now(timezone.utc).isoformat()}), 200

        try:
            regimes = []
            for key, inst in _trader.instrument_map.items():
                try:
                    candles = _trader.client.get_candles(inst.symbol, inst.timeframe, count=200)
                    if candles:
                        regime_state = _regime_detector.update(candles)
                        regimes.append({
                            "instrument": key,
                            "regime": regime_state.regime if hasattr(regime_state, 'regime') else "UNKNOWN",
                            "confidence": float(regime_state.confidence) if hasattr(regime_state, 'confidence') else 0,
                            "adx": float(regime_state.adx_value) if hasattr(regime_state, 'adx_value') else 0,
                            "atr_percentile": float(regime_state.atr_percentile) if hasattr(regime_state, 'atr_percentile') else 0,
                            "trend_strength": float(regime_state.trend_strength) if hasattr(regime_state, 'trend_strength') else 0,
                            "volatility_state": regime_state.volatility_state if hasattr(regime_state, 'volatility_state') else "NORMAL",
                            "recommended_strategy": regime_state.recommended_strategy if hasattr(regime_state, 'recommended_strategy') else "",
                        })
                except Exception as e:
                    logger.warning(f"Regime detection failed for {key}: {e}")

            return jsonify({"regimes": regimes, "timestamp": datetime.now(timezone.utc).isoformat()}), 200
        except Exception as e:
            logger.warning(f"Error processing regimes: {e}")
            return jsonify({"regimes": [], "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}), 200

    except Exception as e:
        logger.exception("Error in /api/regime")
        return jsonify({"regimes": [], "error": str(e)}), 500


@app.route("/api/ml_stats")
def api_ml_stats():
    """ML scorer statistics."""
    try:
        if not _ml_scorer:
            return jsonify({"accuracy": 0, "trades_scored": 0, "features": [], "last_prediction": "-", "timestamp": datetime.now(timezone.utc).isoformat()}), 200

        try:
            stats = _ml_scorer.get_stats()
            importance = _ml_scorer.get_feature_importance()
            features_list = []
            if importance:
                sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
                features_list = [{"name": k, "importance": float(v)} for k, v in sorted_features[:5]]

            accuracy = float(stats.get("model_accuracy", 0))
            completed = int(stats.get("completed_trades", 0))
            trained = bool(stats.get("model_trained", False))

            return jsonify({
                "accuracy": accuracy, "trades_scored": completed, "features": features_list,
                "last_prediction": f"Trained on {completed} trades" if trained else "Not yet trained",
                "model_trained": trained, "wins": int(stats.get("wins", 0)), "losses": int(stats.get("losses", 0)),
                "xgboost_available": stats.get("xgboost_available", False), "model_type": stats.get("model_type", "N/A"),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 200
        except Exception as e:
            logger.warning(f"Error getting ML stats: {e}")
            return jsonify({"accuracy": 0, "trades_scored": 0, "features": [], "last_prediction": f"Error: {e}", "timestamp": datetime.now(timezone.utc).isoformat()}), 200

    except Exception as e:
        logger.exception("Error in /api/ml_stats")
        return jsonify({"error": str(e)}), 500


@app.route("/api/rl_stats")
def api_rl_stats():
    """DQN Reinforcement Learning scorer statistics."""
    try:
        if not _rl_scorer:
            return jsonify({"available": False, "message": "DQN RL Scorer not initialized"}), 200
        stats = _rl_scorer.get_stats()
        return jsonify({
            "available": True, "total_steps": stats.get('total_steps', 0),
            "epsilon": stats.get('epsilon', 1.0), "buffer_size": stats.get('buffer_size', 0),
            "avg_reward": stats.get('avg_reward_50', 0), "total_takes": stats.get('total_takes', 0),
            "total_skips": stats.get('total_skips', 0), "take_rate_pct": stats.get('take_rate_pct', 0),
            "profitable_takes": stats.get('profitable_takes', 0), "avoided_losses": stats.get('avoided_losses', 0),
            "recent_decisions": stats.get('recent_decisions', [])[-5:],
        }), 200
    except Exception as e:
        logger.error(f"RL stats error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/streaming")
def api_streaming():
    """OANDA streaming connection statistics."""
    try:
        if not _stream_manager:
            return jsonify({"available": False, "message": "Streaming not initialized"}), 200
        stats = _stream_manager.get_stats()
        stream_stats = stats.get('stream', {})
        return jsonify({
            "available": True, "connected": stream_stats.get('running', False),
            "total_ticks": stream_stats.get('total_ticks', 0), "ticks_per_second": stream_stats.get('ticks_per_second', 0),
            "uptime_seconds": stream_stats.get('uptime_seconds', 0), "connection_drops": stream_stats.get('connection_drops', 0),
            "instruments": stream_stats.get('instruments_streaming', []),
            "current_spreads": stream_stats.get('current_spreads', {}),
            "spread_violations": stats.get('spread_violations', 0), "flash_crash_alerts": stats.get('flash_crash_alerts', 0),
        }), 200
    except Exception as e:
        logger.error(f"Streaming stats error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/advisor")
def api_advisor():
    """Financial advisor recommendation."""
    try:
        if not _financial_advisor:
            return jsonify({"recommendation": "NORMAL", "reason": "Advisor not initialized", "daily_summary": "No data yet", "timestamp": datetime.now(timezone.utc).isoformat()}), 200
        try:
            rec = _financial_advisor.get_recommendation()
            daily = _financial_advisor.get_daily_summary()
            hours = _financial_advisor.get_best_trading_hours()
            action = rec.get("action", "NORMAL") if isinstance(rec, dict) else "NORMAL"
            reasons = rec.get("reasons", []) if isinstance(rec, dict) else []
            reason_str = "; ".join(reasons) if reasons else "All systems nominal"
            daily_str = daily if isinstance(daily, str) else str(daily) if daily else "No trades today"
            return jsonify({"recommendation": action, "reason": reason_str, "daily_summary": daily_str, "best_hours": hours if isinstance(hours, dict) else {}, "timestamp": datetime.now(timezone.utc).isoformat()}), 200
        except Exception as e:
            logger.warning(f"Error getting advisor recommendation: {e}")
            return jsonify({"recommendation": "NORMAL", "reason": f"Error: {e}", "daily_summary": "-", "timestamp": datetime.now(timezone.utc).isoformat()}), 200
    except Exception as e:
        logger.exception("Error in /api/advisor")
        return jsonify({"error": str(e)}), 500


@app.route("/api/risk")
def api_risk():
    """Risk metrics."""
    try:
        if not _smart_risk:
            return jsonify({"multiplier": 1.0, "current_drawdown": 0.0, "position_size": 0.0, "max_drawdown_limit": 10.0, "risk_assessment": "HEALTHY", "timestamp": datetime.now(timezone.utc).isoformat()}), 200
        try:
            report = _smart_risk.get_risk_report()
            return jsonify({
                "multiplier": 1.0 - (report.get("max_drawdown_pct", 0) / 100.0),
                "current_drawdown": float(report.get("max_drawdown_pct", 0)),
                "position_size": float(report.get("current_balance", 0)),
                "max_drawdown_limit": 10.0, "risk_assessment": report.get("risk_assessment", "HEALTHY"),
                "sharpe_ratio": float(report.get("sharpe_ratio", 0)),
                "concurrent_risk": float(report.get("concurrent_risk_pct", 0)),
                "recent_adjustments": report.get("recent_adjustments", []),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 200
        except Exception as e:
            logger.warning(f"Error getting risk report: {e}")
            return jsonify({"multiplier": 1.0, "current_drawdown": 0.0, "position_size": 0.0, "max_drawdown_limit": 10.0, "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}), 200
    except Exception as e:
        logger.exception("Error in /api/risk")
        return jsonify({"error": str(e)}), 500


@app.route("/api/active_trades")
def api_active_trades():
    """Open positions."""
    try:
        if not _trader or not _trader.client:
            return jsonify({"trades": [], "timestamp": datetime.now(timezone.utc).isoformat()}), 200
        try:
            open_trades_oanda = _trader.client.get_open_trades()
            if not open_trades_oanda:
                return jsonify({"trades": [], "timestamp": datetime.now(timezone.utc).isoformat()}), 200
            price_cache = {}
            formatted_trades = []
            for trade in open_trades_oanda:
                trade_id = trade.get("id", "")
                instrument = trade.get("instrument", "")
                initial_units = float(trade.get("initialUnits", 0))
                direction = "LONG" if initial_units > 0 else "SHORT"
                entry_price = float(trade.get("price", 0))
                unrealized_pl = float(trade.get("unrealizedPL", 0))
                if instrument not in price_cache:
                    try:
                        p = _trader.client.get_current_price(instrument)
                        price_cache[instrument] = p["mid"] if p else entry_price
                    except:
                        price_cache[instrument] = entry_price
                current_price = price_cache[instrument]
                sl_order = trade.get("stopLossOrder", {})
                tp_order = trade.get("takeProfitOrder", {})
                stop_loss = float(sl_order.get("price", 0)) if sl_order else 0
                take_profit = float(tp_order.get("price", 0)) if tp_order else 0
                pnl_pct = (unrealized_pl / (abs(initial_units) * entry_price) * 100) if entry_price > 0 and initial_units != 0 else 0
                formatted_trades.append({
                    "id": trade_id, "instrument": instrument, "direction": direction,
                    "entry_price": entry_price, "current_price": current_price,
                    "stop_loss": stop_loss, "take_profit": take_profit,
                    "pnl": unrealized_pl, "pnl_pct": pnl_pct,
                    "units": int(initial_units), "open_time": trade.get("openTime", ""),
                })
            return jsonify({"trades": formatted_trades, "timestamp": datetime.now(timezone.utc).isoformat()}), 200
        except Exception as e:
            logger.warning(f"Error getting open trades: {e}")
            return jsonify({"trades": [], "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}), 200
    except Exception as e:
        logger.exception("Error in /api/active_trades")
        return jsonify({"trades": [], "error": str(e)}), 500


@app.route("/api/equity_curve")
def api_equity_curve():
    """Equity curve points."""
    try:
        if not _trader:
            return jsonify({"points": [], "timestamp": datetime.now(timezone.utc).isoformat()}), 200
        try:
            equity_history = _trader.equity_history if hasattr(_trader, 'equity_history') else []
            points = [{"time": point.get("time", ""), "balance": float(point.get("balance", 0)), "nav": float(point.get("nav", 0))} for point in equity_history[-500:]]
            return jsonify({"points": points, "timestamp": datetime.now(timezone.utc).isoformat()}), 200
        except Exception as e:
            logger.warning(f"Error getting equity curve: {e}")
            return jsonify({"points": [], "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}), 200
    except Exception as e:
        logger.exception("Error in /api/equity_curve")
        return jsonify({"points": [], "error": str(e)}), 500


@app.route("/api/performance")
def api_performance():
    """Performance metrics."""
    try:
        if not _trader:
            return jsonify({"win_rate": 0.001, "profit_factor": 0, "sharpe": 0, "max_drawdown": 0, "avg_win": 0, "avg_loss": 0, "total_trades": 0, "timestamp": datetime.now(timezone.utc).isoformat()}), 200
        try:
            rm_stats = _trader.risk_mgr.get_stats() if _trader.risk_mgr else {}
            win_rate = float(rm_stats.get("win_rate", 0))
            avg_win = float(rm_stats.get("avg_win", 0))
            avg_loss = float(rm_stats.get("avg_loss", 0))
            total_trades = int(rm_stats.get("total_trades", 0))
            profit_factor = 0
            if _trader.risk_mgr and _trader.risk_mgr.trade_results:
                wins_sum = sum(r["pnl"] for r in _trader.risk_mgr.trade_results if r["pnl"] > 0)
                losses_sum = abs(sum(r["pnl"] for r in _trader.risk_mgr.trade_results if r["pnl"] < 0))
                profit_factor = (wins_sum / losses_sum) if losses_sum > 0 else 0
            sharpe = 0
            if _smart_risk:
                try:
                    report = _smart_risk.get_risk_report()
                    sharpe = float(report.get("sharpe_ratio", 0))
                except:
                    pass
            max_dd = 0
            if _trader.peak_balance and _trader.peak_balance > 0:
                try:
                    acct = _trader.client.get_account_summary()
                    if acct and "account" in acct:
                        bal = float(acct["account"]["balance"])
                        max_dd = max(0, (1 - bal / _trader.peak_balance) * 100)
                except:
                    pass
            return jsonify({
                "win_rate": max(win_rate, 0.001) if total_trades > 0 else 0.001,
                "profit_factor": profit_factor, "sharpe": sharpe, "max_drawdown": max_dd,
                "avg_win": avg_win, "avg_loss": avg_loss, "total_trades": total_trades,
                "consecutive_wins": int(rm_stats.get("consecutive_wins", 0)),
                "consecutive_losses": int(rm_stats.get("consecutive_losses", 0)),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 200
        except Exception as e:
            logger.warning(f"Error getting performance stats: {e}")
            return jsonify({"win_rate": 0.001, "profit_factor": 0, "sharpe": 0, "max_drawdown": 0, "avg_win": 0, "avg_loss": 0, "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}), 200
    except Exception as e:
        logger.exception("Error in /api/performance")
        return jsonify({"error": str(e)}), 500


@app.route("/api/signals")
def api_signals():
    """Recent trading signals."""
    try:
        if not _trader:
            return jsonify({"signals": [], "timestamp": datetime.now(timezone.utc).isoformat()}), 200
        try:
            signals = _trader.recent_signals if hasattr(_trader, 'recent_signals') else []
            formatted_signals = [
                {"time": s.get("time", ""), "instrument": s.get("instrument", ""), "direction": s.get("direction", ""),
                 "strength": float(s.get("strength", 0)), "regime": s.get("regime", ""),
                 "ml_score": float(s.get("ml_score", 0) or 0), "taken": bool(s.get("taken", False)),
                 "reasons": s.get("reasons", [])}
                for s in signals[-10:]
            ]
            return jsonify({"signals": formatted_signals, "timestamp": datetime.now(timezone.utc).isoformat()}), 200
        except Exception as e:
            logger.warning(f"Error getting signals: {e}")
            return jsonify({"signals": [], "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}), 200
    except Exception as e:
        logger.exception("Error in /api/signals")
        return jsonify({"signals": [], "error": str(e)}), 500


@app.route("/api/trades_history")
def api_trades_history():
    """Closed trades history."""
    try:
        if not _trader:
            return jsonify({"trades": [], "timestamp": datetime.now(timezone.utc).isoformat()}), 200
        try:
            closed_trades = _trader.closed_trades_log if hasattr(_trader, 'closed_trades_log') else []
            formatted_trades = [
                {"instrument": t.get("symbol", ""), "direction": t.get("direction", ""),
                 "entry_price": float(t.get("entry", 0)), "exit_price": float(t.get("entry", 0)),
                 "pnl": float(t.get("pnl_atr", 0)), "bars": 0,
                 "exit_type": "TP" if float(t.get("pnl_atr", 0)) > 0 else "SL",
                 "trade_id": t.get("trade_id", ""), "time": t.get("time", "")}
                for t in closed_trades[-50:]
            ]
            return jsonify({"trades": formatted_trades, "timestamp": datetime.now(timezone.utc).isoformat()}), 200
        except Exception as e:
            logger.warning(f"Error getting closed trades: {e}")
            return jsonify({"trades": [], "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}), 200
    except Exception as e:
        logger.exception("Error in /api/trades_history")
        return jsonify({"trades": [], "error": str(e)}), 500


@app.route("/api/optimization")
def api_optimization():
    """Walk-forward optimization status."""
    try:
        if not _walk_forward:
            return jsonify({"last_opt_time": None, "is_results": {}, "oos_results": {}, "params_adopted": {}, "timestamp": datetime.now(timezone.utc).isoformat()}), 200
        try:
            opt_history = _walk_forward.get_optimization_history()
            if opt_history and len(opt_history) > 0:
                last = opt_history[-1]
                return jsonify({
                    "last_opt_time": last.timestamp if hasattr(last, 'timestamp') else None,
                    "is_results": {"sharpe": float(last.is_pf) if hasattr(last, 'is_pf') else 0, "win_rate": float(last.is_wr) if hasattr(last, 'is_wr') else 0, "trades": int(last.is_trades) if hasattr(last, 'is_trades') else 0},
                    "oos_results": {"sharpe": float(last.oos_pf) if hasattr(last, 'oos_pf') else 0, "win_rate": float(last.oos_wr) if hasattr(last, 'oos_wr') else 0, "trades": int(last.oos_trades) if hasattr(last, 'oos_trades') else 0},
                    "params_adopted": last.params if hasattr(last, 'params') else {},
                    "adopted": bool(last.adopted) if hasattr(last, 'adopted') else False,
                    "total_optimizations": len(opt_history),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }), 200
            else:
                return jsonify({"last_opt_time": None, "is_results": {}, "oos_results": {}, "params_adopted": {}, "adopted": False, "total_optimizations": 0, "timestamp": datetime.now(timezone.utc).isoformat()}), 200
        except Exception as e:
            logger.warning(f"Error getting optimization history: {e}")
            return jsonify({"last_opt_time": None, "is_results": {}, "oos_results": {}, "params_adopted": {}, "timestamp": datetime.now(timezone.utc).isoformat()}), 200
    except Exception as e:
        logger.exception("Error in /api/optimization")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v4_status")
def api_v4_status():
    """Combined status of all v4 modules."""
    try:
        return jsonify({
            "rl_scorer": {"available": _rl_scorer is not None, "epsilon": float(_rl_scorer.get_stats().get('epsilon', 1.0)) if _rl_scorer else None, "total_steps": int(_rl_scorer.get_stats().get('total_steps', 0)) if _rl_scorer else 0},
            "streaming": {"available": _stream_manager is not None, "connected": _stream_manager.get_stats().get('stream', {}).get('running', False) if _stream_manager else False},
            "xgboost": {"available": _ml_scorer.get_stats().get('xgboost_available', False) if _ml_scorer else False, "active": _ml_scorer.get_stats().get('model_type', 'N/A') == 'xgboost' if _ml_scorer else False},
            "genetic_algorithm": {"available": _walk_forward is not None, "has_optimizer": hasattr(_walk_forward, 'run_genetic_optimization') if _walk_forward else False},
            "emergency_brake": {"available": _smart_risk is not None},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        logger.error(f"V4 status error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    """Run a visual backtest."""
    try:
        from backtester import Backtester
        data = request.get_json(force=True) or {}
        symbol = data.get("symbol", "EUR_USD")
        timeframe = data.get("timeframe", "M30")
        lookback_days = int(data.get("lookback_days", 30))
        risk_pct = float(data.get("risk_pct", 0.02))
        if not _trader or not _trader.client:
            return jsonify({"error": "Trader not initialized"}), 500
        bt = Backtester(_trader.client)
        results = bt.run_visual(symbol, timeframe, lookback_days=lookback_days, risk_per_trade=risk_pct)
        return jsonify(results), 200
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v5_status")
def api_v5_status():
    """Status of v5 modules: Correlation, Session, Ensemble, LSTM, Anomaly."""
    try:
        status = {"timestamp": datetime.now(timezone.utc).isoformat()}
        if _trader:
            status["correlation"] = {"available": _trader.correlation_mgr is not None, "active_exposures": len(_trader.correlation_mgr.active_exposures) if _trader.correlation_mgr else 0, "group_status": _trader.correlation_mgr.get_group_status() if _trader.correlation_mgr else []}
            status["session_filter"] = {"available": _trader.session_filter is not None, "dashboard": _trader.session_filter.get_dashboard_status() if _trader.session_filter else {}}
            status["ensemble"] = {"available": _trader.ensemble_scorer is not None, "status": _trader.ensemble_scorer.get_status() if _trader.ensemble_scorer else {}}
            status["lstm"] = {"available": _trader.lstm_predictor is not None, "status": _trader.lstm_predictor.get_status() if _trader.lstm_predictor else {}}
            status["anomaly_detector"] = {"available": _trader.anomaly_detector is not None, "status": _trader.anomaly_detector.get_status() if _trader.anomaly_detector else {}}
            status["crypto"] = {"clients": list(_trader.crypto_clients.keys()) if hasattr(_trader, 'crypto_clients') else []}
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"V5 status error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/gpu_status")
def api_gpu_status():
    """GPU stats."""
    try:
        status = {"cuda_available": False, "device": "cpu"}
        if _trader and _trader.lstm_predictor:
            status = _trader.lstm_predictor.get_gpu_status()
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"GPU status error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/market_enrichment")
def api_market_enrichment():
    """External market data: VIX, DXY, correlations, macro."""
    try:
        data = {"available": False}
        if _trader and hasattr(_trader, 'market_enricher') and _trader.market_enricher:
            data = {"available": True, "context": _trader.market_enricher.get_market_context(), "status": _trader.market_enricher.get_status()}
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Market enrichment error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/news_sentiment")
def api_news_sentiment():
    """News feed and sentiment analysis."""
    try:
        data = {"available": False}
        if _trader and hasattr(_trader, 'news_feed') and _trader.news_feed:
            data = {"available": True, "overview": _trader.news_feed.get_market_overview(), "status": _trader.news_feed.get_status()}
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"News sentiment error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/advisor_es")
def api_advisor_es():
    """Financial advisor data (Spanish agent)."""
    try:
        data = {}
        if _trader and hasattr(_trader, 'advisor') and _trader.advisor:
            data = _trader.advisor.get_dashboard_data()
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Advisor ES error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/advanced_learning")
def api_advanced_learning():
    """Advanced learning module status."""
    try:
        data = {"available": False}
        if _trader:
            data["available"] = True
            modules = {}
            lstm = getattr(_trader, 'lstm_predictor', None)
            if lstm:
                if hasattr(lstm, 'last_uncertainty') and lstm.last_uncertainty:
                    modules["mc_dropout"] = lstm.last_uncertainty
                if hasattr(lstm, 'last_quantiles') and lstm.last_quantiles:
                    modules["quantile_regression"] = lstm.last_quantiles
                if hasattr(lstm, 'mc_dropout') and lstm.mc_dropout:
                    modules["mc_dropout_config"] = lstm.mc_dropout.get_status()
                adv = {}
                if hasattr(lstm, 'sharpe_loss'): adv["sharpe_loss"] = lstm.sharpe_loss is not None
                if hasattr(lstm, 'augmentor'): adv["data_augmentation"] = lstm.augmentor is not None
                if hasattr(lstm, 'curriculum'): adv["curriculum_learning"] = lstm.curriculum is not None
                modules["lstm_advanced"] = adv
            hmm = getattr(_trader, 'hmm_regime', None)
            if hmm: modules["hmm_regime"] = hmm.get_regime()
            fisher = getattr(_trader, 'fisher_detector', None)
            if fisher: modules["fisher_info"] = fisher.get_status()
            tm = getattr(_trader, 'training_manager', None)
            if tm:
                if hasattr(tm, 'td_evaluator') and tm.td_evaluator: modules["td_lambda"] = tm.td_evaluator.get_status()
                if hasattr(tm, 'fisher_detector') and tm.fisher_detector: modules["training_fisher"] = tm.fisher_detector.get_status()
            ens = getattr(_trader, 'ensemble_scorer', None)
            if ens and hasattr(ens, 'exp3') and ens.exp3: modules["exp3_online"] = ens.exp3.get_status()
            if lstm and hasattr(lstm, 'last_wavelet_energy') and lstm.last_wavelet_energy: modules["wavelet_energy"] = lstm.last_wavelet_energy
            if lstm and hasattr(lstm, 'last_drift_status') and lstm.last_drift_status: modules["wasserstein_drift"] = lstm.last_drift_status
            if tm and hasattr(tm, 'granger_selector') and tm.granger_selector: modules["granger_causality"] = tm.granger_selector.get_status()
            if tm and hasattr(tm, 'wasserstein_monitor') and tm.wasserstein_monitor: modules["wasserstein_training"] = tm.wasserstein_monitor.get_status()
            vae = getattr(_trader, 'market_vae', None)
            if vae: modules["market_vae"] = {"available": True}
            gnn = getattr(_trader, 'cross_asset_gnn', None)
            if gnn: modules["cross_asset_gnn"] = {"available": True, "n_assets": 8}
            data["modules"] = modules
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Advanced learning status error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/training_status")
def api_training_status():
    """Training manager status."""
    try:
        data = {"available": False, "last_nightly": None, "last_weekly": None, "nightly_results": [], "weekly_reports": [], "adaptive_config": {}, "next_training": None, "next_review": None}
        if _trader and hasattr(_trader, 'training_manager') and _trader.training_manager:
            tm = _trader.training_manager
            data["available"] = True
            data["last_nightly"] = tm.last_nightly_date
            data["last_weekly"] = tm.last_weekly_date
            data["adaptive_config"] = getattr(tm, 'adaptive_config', {})
            import glob
            log_dir = getattr(tm, 'log_dir', 'training_logs')
            nightly_files = sorted(glob.glob(os.path.join(log_dir, 'nightly_*.json')))[-7:]
            for f in nightly_files:
                try:
                    with open(f) as fh: data["nightly_results"].append(json.load(fh))
                except: pass
            weekly_files = sorted(glob.glob(os.path.join(log_dir, 'weekly_*.json')))[-4:]
            for f in weekly_files:
                try:
                    with open(f) as fh: data["weekly_reports"].append(json.load(fh))
                except: pass
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Training status error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/telegram_chat", methods=["POST"])
def api_telegram_chat():
    """Process a chat command from the dashboard."""
    try:
        data = request.get_json(force=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"reply": "Please type a command or question."}), 200
        cmd = message.split()[0].lower() if message.startswith("/") else ""
        if cmd == "/status" or "status" in message.lower(): return _chat_status()
        elif cmd == "/trades" or "trades" in message.lower() or "open" in message.lower(): return _chat_trades()
        elif cmd == "/pnl" or "pnl" in message.lower() or "profit" in message.lower(): return _chat_pnl()
        elif cmd == "/risk" or "risk" in message.lower(): return _chat_risk()
        elif cmd == "/regime" or "regime" in message.lower() or "market" in message.lower(): return _chat_regime()
        elif cmd == "/streaming" or "stream" in message.lower(): return _chat_streaming()
        elif cmd == "/rl" or "rl" in message.lower() or "dqn" in message.lower(): return _chat_rl()
        elif cmd == "/help" or cmd == "/start": return _chat_help()
        else: return _chat_help()
    except Exception as e:
        logger.error(f"Telegram chat error: {e}")
        return jsonify({"reply": f"Error: {e}"}), 200


def _chat_status():
    try:
        if not _trader or not _trader.client:
            return jsonify({"reply": "Bot not connected"}), 200
        account_data = _trader.client.get_account_summary()
        acct = account_data.get("account", {}) if account_data else {}
        balance = float(acct.get("balance", 0))
        nav = float(acct.get("NAV", 0))
        unrealized = float(acct.get("unrealizedPL", 0))
        open_count = int(acct.get("openTradeCount", 0))
        state = "Running" if _trader._running else "Stopped"
        rm_stats = _trader.risk_mgr.get_stats() if _trader.risk_mgr else {}
        wr = float(rm_stats.get("win_rate", 0))
        total_trades = int(rm_stats.get("total_trades", 0))
        pf = 0
        if _trader.risk_mgr and _trader.risk_mgr.trade_results:
            ws = sum(r["pnl"] for r in _trader.risk_mgr.trade_results if r["pnl"] > 0)
            ls = abs(sum(r["pnl"] for r in _trader.risk_mgr.trade_results if r["pnl"] < 0))
            pf = (ws / ls) if ls > 0 else 0
        reply = f"BOT STATUS\n\nState: {state}\nOpen Trades: {open_count}\n\nBalance: ${balance:,.2f}\nNAV: ${nav:,.2f}\nUnrealized: ${unrealized:+,.2f}\n\nClosed Trades: {total_trades}\nWin Rate: {wr:.1%}\nProfit Factor: {pf:.2f}"
        return jsonify({"reply": reply}), 200
    except Exception as e:
        return jsonify({"reply": f"Error: {e}"}), 200


def _chat_trades():
    try:
        if not _trader or not _trader.client:
            return jsonify({"reply": "Bot not connected"}), 200
        trades = _trader.client.get_open_trades() or []
        if not trades:
            return jsonify({"reply": "No open trades"}), 200
        msg = f"OPEN TRADES ({len(trades)})\n\n"
        for t in trades:
            inst = t.get("instrument", "")
            units = int(float(t.get("currentUnits", 0)))
            entry = float(t.get("price", 0))
            pnl = float(t.get("unrealizedPL", 0))
            d = "LONG" if units > 0 else "SHORT"
            prec = 3 if "JPY" in inst else 5
            msg += f"{d} {inst}\n  Entry: {entry:.{prec}f} | P&L: ${pnl:+.2f}\n\n"
        return jsonify({"reply": msg}), 200
    except Exception as e:
        return jsonify({"reply": f"Error: {e}"}), 200


def _chat_pnl():
    try:
        if not _trader or not _trader.client:
            return jsonify({"reply": "Bot not connected"}), 200
        account_data = _trader.client.get_account_summary()
        acct = account_data.get("account", {}) if account_data else {}
        unrealized = float(acct.get("unrealizedPL", 0))
        todays = float(acct.get("pl", 0))
        rm_stats = _trader.risk_mgr.get_stats() if _trader.risk_mgr else {}
        total_trades = int(rm_stats.get("total_trades", 0))
        avg_win = float(rm_stats.get("avg_win", 0))
        avg_loss = float(rm_stats.get("avg_loss", 0))
        max_dd = 0
        if _trader.peak_balance and _trader.peak_balance > 0:
            max_dd = max(0, (1 - float(acct.get("balance", 0)) / _trader.peak_balance) * 100)
        total_pnl = todays + unrealized
        reply = f"P&L REPORT\n\nToday: ${todays:+,.2f}\nUnrealized: ${unrealized:+,.2f}\nCombined: ${total_pnl:+,.2f}\n\nTotal Trades: {total_trades}\nAvg Win: ${avg_win:,.2f}\nAvg Loss: ${avg_loss:,.2f}\nMax DD: {max_dd:.1f}%"
        return jsonify({"reply": reply}), 200
    except Exception as e:
        return jsonify({"reply": f"Error: {e}"}), 200


def _chat_risk():
    try:
        if not _smart_risk:
            return jsonify({"reply": "Risk module not available"}), 200
        report = _smart_risk.get_risk_report()
        dd = float(report.get("max_drawdown_pct", 0))
        mult = 1.0 - (dd / 100.0)
        reply = f"RISK STATUS\n\nAssessment: {report.get('risk_assessment', 'HEALTHY')}\nMultiplier: {mult:.2f}x\nDrawdown: {dd:.1f}%\nMax DD Limit: 10.0%\nSharpe: {float(report.get('sharpe_ratio', 0)):.2f}\nConcurrent Risk: {float(report.get('concurrent_risk_pct', 0)):.1f}%"
        return jsonify({"reply": reply}), 200
    except Exception as e:
        return jsonify({"reply": f"Error: {e}"}), 200


def _chat_regime():
    try:
        if not _regime_detector or not _trader or not _trader.client:
            return jsonify({"reply": "No regime data available yet"}), 200
        regimes = []
        for key, inst in _trader.instrument_map.items():
            try:
                candles = _trader.client.get_candles(inst.symbol, inst.timeframe, count=200)
                if candles:
                    rs = _regime_detector.update(candles)
                    regimes.append({"instrument": key, "regime": rs.regime if hasattr(rs, 'regime') else "?", "adx": float(rs.adx_value) if hasattr(rs, 'adx_value') else 0, "confidence": float(rs.confidence) if hasattr(rs, 'confidence') else 0})
            except Exception: pass
        if not regimes:
            return jsonify({"reply": "No regime data available yet"}), 200
        msg = "MARKET REGIMES\n\n"
        for r in regimes:
            msg += f"{r['instrument']}\n  Regime: {r['regime']}\n  ADX: {r['adx']:.1f} | Confidence: {r['confidence']:.0f}%\n\n"
        return jsonify({"reply": msg}), 200
    except Exception as e:
        return jsonify({"reply": f"Error: {e}"}), 200


def _chat_streaming():
    try:
        if not _stream_manager:
            return jsonify({"reply": "Streaming not available"}), 200
        stats = _stream_manager.get_stats()
        reply = f"STREAMING STATUS\n\nConnected: {stats.get('connected', False)}\nTicks/sec: {stats.get('ticks_per_second',0):.1f}\nTotal ticks: {stats.get('total_ticks',0)}\nUptime: {stats.get('uptime_seconds',0):.0f}s\nDrops: {stats.get('connection_drops',0)}\nSpread violations: {stats.get('spread_violations',0)}\nFlash crash alerts: {stats.get('flash_crash_alerts',0)}"
        spreads = stats.get("current_spreads", {})
        if spreads:
            reply += "\n\nSpreads:\n"
            for inst, sp in spreads.items():
                pip = sp * 10000 if "JPY" not in inst else sp * 100
                reply += f"  {inst}: {pip:.1f} pips\n"
        return jsonify({"reply": reply}), 200
    except Exception as e:
        return jsonify({"reply": f"Error: {e}"}), 200


def _chat_rl():
    try:
        if not _rl_scorer:
            return jsonify({"reply": "RL Scorer not available"}), 200
        stats = _rl_scorer.get_stats()
        reply = f"DQN RL SCORER\n\nEpsilon: {stats.get('epsilon',0):.3f}\nSteps: {stats.get('total_steps',0)}\nBuffer: {stats.get('buffer_size',0)}\nTakes: {stats.get('total_takes',0)} | Skips: {stats.get('total_skips',0)}\nAvg Reward: {stats.get('avg_reward',0):.4f}\nAvoided Losses: {stats.get('avoided_losses',0)}"
        return jsonify({"reply": reply}), 200
    except Exception as e:
        return jsonify({"reply": f"Error: {e}"}), 200


def _chat_help():
    reply = "TRADING ASSISTANT\n\nAvailable commands:\n\n/status - Bot & account status\n/trades - Open positions\n/pnl - Profit & loss report\n/risk - Risk assessment\n/regime - Market regimes\n/streaming - Stream status\n/rl - DQN RL scorer stats\n\nYou can also type keywords like:\n  'status', 'trades', 'pnl', 'risk',\n  'market', 'stream', 'dqn'"
    return jsonify({"reply": reply}), 200


# =====================================================================
# v9 ENDPOINTS
# =====================================================================

@app.route("/api/backtest_status")
def api_backtest_status():
    try:
        data = {"available": False}
        bt = getattr(_trader, 'backtester', None) if _trader else None
        if bt: data["available"] = True; data.update(bt.get_status())
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/smart_alerts")
def api_smart_alerts():
    try:
        data = {"available": False}
        alerts = getattr(_trader, 'smart_alerts', None) if _trader else None
        if alerts: data["available"] = True; data.update(alerts.get_status())
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/hyperopt_status")
def api_hyperopt_status():
    try:
        data = {"available": False}
        opt = getattr(_trader, 'hyperopt', None) if _trader else None
        if opt: data["available"] = True; data.update(opt.get_status())
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/mtf_status")
def api_mtf_status():
    try:
        data = {"available": False}
        mtf = getattr(_trader, 'mtf_lstm', None) if _trader else None
        if mtf: data["available"] = True; data.update(mtf.get_status())
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/v9_overview")
def api_v9_overview():
    try:
        data = {
            "backtester": getattr(_trader, 'backtester', None) is not None if _trader else False,
            "smart_alerts": getattr(_trader, 'smart_alerts', None) is not None if _trader else False,
            "hyperopt": getattr(_trader, 'hyperopt', None) is not None if _trader else False,
            "mtf_lstm": getattr(_trader, 'mtf_lstm', None) is not None if _trader else False,
        }
        alerts = getattr(_trader, 'smart_alerts', None) if _trader else None
        if alerts: data["alerts_today"] = alerts._alerts_today; data["recent_alerts"] = alerts.get_recent_alerts(5)
        mtf = getattr(_trader, 'mtf_lstm', None) if _trader else None
        if mtf: data["tf_importance"] = mtf.last_tf_importance
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== v10 ENDPOINTS =====

@app.route("/api/kelly_status")
def api_kelly_status():
    try:
        kelly = getattr(_trader, 'kelly_sizer', None) if _trader else None
        if not kelly: return jsonify({"available": False}), 200
        return jsonify({"available": True, **kelly.get_status()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/walk_forward_status")
def api_walk_forward_status():
    try:
        wfv = getattr(_trader, 'wf_validator', None) if _trader else None
        if not wfv: return jsonify({"available": False}), 200
        return jsonify({"available": True, **wfv.get_status()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/experiments")
def api_experiments():
    try:
        tracker = getattr(_trader, 'experiment_tracker', None) if _trader else None
        if not tracker: return jsonify({"available": False}), 200
        status = tracker.get_status()
        leaderboard = tracker.get_leaderboard("sharpe_ratio", 10)
        return jsonify({
            "available": True, "total": status["total_experiments"], "latest": status.get("latest"),
            "leaderboard_sharpe": [{"id": e["id"], "sharpe": e["metrics"].get("sharpe_ratio", 0), "pf": e["metrics"].get("profit_factor", 0), "win_rate": e["metrics"].get("win_rate", 0), "timestamp": e.get("timestamp", "")} for e in leaderboard],
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/portfolio_status")
def api_portfolio_status():
    try:
        po = getattr(_trader, 'portfolio_optimizer', None) if _trader else None
        if not po: return jsonify({"available": False}), 200
        return jsonify({"available": True, **po.get_status()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/v10_overview")
def api_v10_overview():
    try:
        data = {
            "kelly": getattr(_trader, 'kelly_sizer', None) is not None if _trader else False,
            "walk_forward": getattr(_trader, 'wf_validator', None) is not None if _trader else False,
            "experiments": getattr(_trader, 'experiment_tracker', None) is not None if _trader else False,
            "portfolio": getattr(_trader, 'portfolio_optimizer', None) is not None if _trader else False,
        }
        kelly = getattr(_trader, 'kelly_sizer', None) if _trader else None
        if kelly: data["kelly_size_pct"] = kelly.last_size_pct; data["kelly_raw"] = kelly.raw_kelly; data["kelly_win_rate"] = kelly.win_rate
        tracker = getattr(_trader, 'experiment_tracker', None) if _trader else None
        if tracker: data["total_experiments"] = tracker.get_status()["total_experiments"]
        po = getattr(_trader, 'portfolio_optimizer', None) if _trader else None
        if po: data["portfolio_weights"] = po.last_weights; data["portfolio_metrics"] = po.last_metrics
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard")
def serve_dashboard_v2():
    """Serve the interactive HTML dashboard."""
    import os
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_v2.html")
    return send_file(html_path, mimetype='text/html')


def run_dashboard():
    """Start the Flask dashboard server."""
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False)


if __name__ == "__main__":
    run_dashboard()
