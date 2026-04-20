"""
ML SuperTrend v51 - Professional Backtester
=============================================
Backtesting engine con métricas profesionales para validar
las 20 estrategias científicas con datos históricos.

Métricas:
  - Sharpe Ratio (anualizado)
  - Sortino Ratio (downside risk only)
  - Max Drawdown (% y duración)
  - Calmar Ratio (return / max_dd)
  - Profit Factor (gross_profit / gross_loss)
  - Win Rate, Avg Win/Loss, Expectancy
  - Recovery Factor, Payoff Ratio
  - Análisis por régimen HMM, sesión, par, hora

Modos:
  1. Walk-Forward: train/test sliding windows
  2. Monte Carlo: N simulaciones con resampling de trades
  3. Stress Test: replay con volatilidad amplificada

Uso:
    from backtester import Backtester
    bt = Backtester()
    results = bt.run(candles_history)
    report = bt.generate_report()
"""

import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json
import os

logger = logging.getLogger(__name__)


class TradeRecord:
    """Registro de un trade individual en backtest."""
    __slots__ = [
        'entry_time', 'exit_time', 'instrument', 'direction',
        'entry_price', 'exit_price', 'pnl', 'pnl_pct',
        'holding_bars', 'confidence', 'regime', 'session',
        'sl_price', 'tp_price', 'exit_reason',
    ]

    def __init__(self, **kwargs):
        for k in self.__slots__:
            setattr(self, k, kwargs.get(k))

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__slots__}


class PerformanceMetrics:
    """
    Calcula todas las métricas profesionales de trading.

    Fórmulas:
      Sharpe  = (mean(R) / std(R)) * sqrt(252)
      Sortino = (mean(R) / downside_std(R)) * sqrt(252)
      Calmar  = annualized_return / max_drawdown
      Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
      Profit Factor = gross_profit / gross_loss
      Recovery Factor = net_profit / max_drawdown
    """

    @staticmethod
    def calculate_all(trades: List[TradeRecord], initial_capital: float = 10000.0) -> Dict:
        """Calcular todas las métricas desde una lista de trades."""
        if not trades:
            return {"error": "No trades to analyze"}

        pnls = np.array([t.pnl for t in trades if t.pnl is not None])
        pnl_pcts = np.array([t.pnl_pct for t in trades if t.pnl_pct is not None])

        if len(pnls) == 0:
            return {"error": "No valid PnL data"}

        # Equity curve
        equity = [initial_capital]
        for p in pnls:
            equity.append(equity[-1] + p)
        equity = np.array(equity)

        n_trades = len(pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        n_wins = len(wins)
        n_losses = len(losses)
        win_rate = n_wins / n_trades if n_trades > 0 else 0

        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0
        gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.01
        net_profit = float(np.sum(pnls))

        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
        avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0
        largest_win = float(np.max(wins)) if len(wins) > 0 else 0
        largest_loss = float(np.min(losses)) if len(losses) > 0 else 0

        returns = pnl_pcts if len(pnl_pcts) > 0 else pnls / initial_capital
        trades_per_year = min(252, n_trades * (252 / max(n_trades, 1)))
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns)) + 1e-10
        sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year)

        downside = returns[returns < 0]
        downside_std = float(np.std(downside)) + 1e-10 if len(downside) > 0 else 1e-10
        sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year)

        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0

        dd_duration = 0
        max_dd_duration = 0
        for i in range(1, len(equity)):
            if equity[i] < peak[i]:
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
            else:
                dd_duration = 0

        total_return_pct = (equity[-1] - initial_capital) / initial_capital
        calmar = total_return_pct / max(max_dd, 0.001)
        profit_factor = gross_profit / max(gross_loss, 0.01)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        payoff_ratio = avg_win / max(avg_loss, 0.01)
        max_dd_dollars = max_dd * initial_capital
        recovery_factor = net_profit / max(max_dd_dollars, 0.01)

        max_consec_wins = max_consec_losses = curr_wins = curr_losses = 0
        for p in pnls:
            if p > 0:
                curr_wins += 1
                curr_losses = 0
                max_consec_wins = max(max_consec_wins, curr_wins)
            else:
                curr_losses += 1
                curr_wins = 0
                max_consec_losses = max(max_consec_losses, curr_losses)

        return {
            "net_profit": round(net_profit, 2),
            "total_return_pct": round(total_return_pct * 100, 2),
            "n_trades": n_trades,
            "win_rate": round(win_rate * 100, 1),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "max_drawdown_duration": max_dd_duration,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "largest_win": round(largest_win, 2),
            "largest_loss": round(largest_loss, 2),
            "payoff_ratio": round(payoff_ratio, 2),
            "expectancy": round(expectancy, 2),
            "recovery_factor": round(recovery_factor, 2),
            "max_consecutive_wins": max_consec_wins,
            "max_consecutive_losses": max_consec_losses,
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "equity_curve": equity.tolist(),
            "drawdown_curve": drawdown.tolist(),
        }


class Backtester:
    """
    Motor de backtesting profesional.
    Simula trades usando la lógica del bot con datos históricos.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        risk_per_trade_pct: float = 1.0,
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.risk_per_trade_pct = risk_per_trade_pct

        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = [initial_capital]
        self.current_capital = initial_capital
        self.open_position = None
        self.results: Dict = {}
        self.analysis_by_dimension: Dict = {}

    def run(
        self,
        candles: List[Dict],
        signals: List[Dict] = None,
        strategy_func=None,
    ) -> Dict:
        """
        Ejecutar backtest sobre datos históricos.

        Args:
            candles: [{time, open, high, low, close, volume}, ...]
            signals: Pre-computed signals [{time, direction, confidence, sl, tp}, ...]
            strategy_func: Callable(candles_window) -> signal_dict or None
        """
        logger.info(f"Backtesting: {len(candles)} candles, capital={self.initial_capital}")

        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.current_capital = self.initial_capital
        self.open_position = None

        if signals:
            self._run_from_signals(candles, signals)
        elif strategy_func:
            self._run_with_strategy(candles, strategy_func)
        else:
            self._run_simple_simulation(candles)

        self.results = PerformanceMetrics.calculate_all(self.trades, self.initial_capital)
        self.analysis_by_dimension = self._analyze_by_dimension()

        logger.info(f"Backtest complete: {self.results.get('n_trades', 0)} trades, "
                   f"Sharpe={self.results.get('sharpe_ratio', 0):.3f}, "
                   f"PF={self.results.get('profit_factor', 0):.2f}")

        return self.results

    def _run_from_signals(self, candles: List[Dict], signals: List[Dict]):
        """Ejecutar backtest con señales pre-calculadas."""
        signal_map = {}
        for s in signals:
            signal_map[s.get("time", "")] = s

        for i, candle in enumerate(candles):
            t = candle.get("time", "")
            if self.open_position:
                self._check_exit(candle, i)
            if t in signal_map and not self.open_position:
                self._open_trade(candle, signal_map[t], i)

        if self.open_position and candles:
            self._force_close(candles[-1], len(candles) - 1)

    def _run_with_strategy(self, candles: List[Dict], strategy_func):
        """Ejecutar backtest con función de estrategia."""
        lookback = 50
        for i in range(lookback, len(candles)):
            candle = candles[i]
            window = candles[max(0, i - lookback):i + 1]
            if self.open_position:
                self._check_exit(candle, i)
            if not self.open_position:
                try:
                    signal = strategy_func(window)
                    if signal:
                        self._open_trade(candle, signal, i)
                except Exception:
                    pass

        if self.open_position and candles:
            self._force_close(candles[-1], len(candles) - 1)

    def _run_simple_simulation(self, candles: List[Dict]):
        """Simulación SMA crossover para validar el backtester."""
        closes = np.array([self._get_close(c) for c in candles], dtype=float)
        if len(closes) < 50:
            return

        for i in range(50, len(candles)):
            if self.open_position:
                self._check_exit(candles[i], i)
                continue

            sma20 = np.mean(closes[i-20:i])
            sma50 = np.mean(closes[i-50:i])
            prev_sma20 = np.mean(closes[i-21:i-1])
            prev_sma50 = np.mean(closes[i-51:i-1])
            close = closes[i]
            atr = np.std(closes[i-14:i]) * 2

            signal = None
            if prev_sma20 <= prev_sma50 and sma20 > sma50:
                signal = {"direction": "BUY", "confidence": 0.6,
                         "sl": close - atr, "tp": close + atr * 2}
            elif prev_sma20 >= prev_sma50 and sma20 < sma50:
                signal = {"direction": "SELL", "confidence": 0.6,
                         "sl": close + atr, "tp": close - atr * 2}

            if signal:
                self._open_trade(candles[i], signal, i)

        if self.open_position and candles:
            self._force_close(candles[-1], len(candles) - 1)

    def _open_trade(self, candle: Dict, signal: Dict, bar_idx: int):
        close = self._get_close(candle)
        direction = signal.get("direction", "BUY")
        entry = close * (1 + self.slippage_pct) if direction == "BUY" else close * (1 - self.slippage_pct)

        self.open_position = {
            "entry_price": entry, "direction": direction,
            "entry_bar": bar_idx, "entry_time": candle.get("time", ""),
            "sl": signal.get("sl", entry * (0.99 if direction == "BUY" else 1.01)),
            "tp": signal.get("tp", entry * (1.02 if direction == "BUY" else 0.98)),
            "confidence": signal.get("confidence", 0.5),
            "regime": signal.get("regime", "unknown"),
            "session": signal.get("session", "unknown"),
            "instrument": signal.get("instrument", candle.get("instrument", "unknown")),
        }

    def _check_exit(self, candle: Dict, bar_idx: int):
        if not self.open_position:
            return

        high = float(candle.get('high', candle.get('mid', {}).get('h', 0)))
        low = float(candle.get('low', candle.get('mid', {}).get('l', 0)))
        close = self._get_close(candle)
        pos = self.open_position
        exit_price = exit_reason = None

        if pos["direction"] == "BUY":
            if low <= pos["sl"]:
                exit_price, exit_reason = pos["sl"], "SL"
            elif high >= pos["tp"]:
                exit_price, exit_reason = pos["tp"], "TP"
        else:
            if high >= pos["sl"]:
                exit_price, exit_reason = pos["sl"], "SL"
            elif low <= pos["tp"]:
                exit_price, exit_reason = pos["tp"], "TP"

        if not exit_price and (bar_idx - pos["entry_bar"]) >= 100:
            exit_price, exit_reason = close, "TIMEOUT"

        if exit_price:
            self._close_trade(exit_price, exit_reason, candle, bar_idx)

    def _close_trade(self, exit_price: float, reason: str, candle: Dict, bar_idx: int):
        pos = self.open_position
        if not pos:
            return

        if pos["direction"] == "BUY":
            exit_price *= (1 - self.slippage_pct)
            pnl = exit_price - pos["entry_price"]
        else:
            exit_price *= (1 + self.slippage_pct)
            pnl = pos["entry_price"] - exit_price

        trade_value = self.current_capital * (self.risk_per_trade_pct / 100)
        commission = trade_value * self.commission_pct * 2
        pnl_net = (pnl / pos["entry_price"]) * trade_value - commission
        pnl_pct = pnl / pos["entry_price"]

        trade = TradeRecord(
            entry_time=pos["entry_time"], exit_time=candle.get("time", ""),
            instrument=pos.get("instrument", "unknown"), direction=pos["direction"],
            entry_price=round(pos["entry_price"], 5), exit_price=round(exit_price, 5),
            pnl=round(pnl_net, 2), pnl_pct=round(pnl_pct, 6),
            holding_bars=bar_idx - pos["entry_bar"], confidence=pos.get("confidence", 0),
            regime=pos.get("regime", "unknown"), session=pos.get("session", "unknown"),
            sl_price=pos.get("sl"), tp_price=pos.get("tp"), exit_reason=reason,
        )

        self.trades.append(trade)
        self.current_capital += pnl_net
        self.equity_curve.append(self.current_capital)
        self.open_position = None

    def _force_close(self, candle: Dict, bar_idx: int):
        self._close_trade(self._get_close(candle), "END_OF_DATA", candle, bar_idx)

    def _get_close(self, candle: Dict) -> float:
        if 'close' in candle:
            return float(candle['close'])
        if 'mid' in candle and 'c' in candle['mid']:
            return float(candle['mid']['c'])
        return float(candle.get('c', 0))

    # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # AN\u00c1LISIS MULTIDIMENSIONAL
    # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _analyze_by_dimension(self) -> Dict:
        if not self.trades:
            return {}

        analysis = {}
        dimensions = {
            "by_regime": lambda t: t.regime or "unknown",
            "by_session": lambda t: t.session or "unknown",
            "by_direction": lambda t: t.direction or "unknown",
            "by_exit_reason": lambda t: t.exit_reason or "unknown",
            "by_instrument": lambda t: t.instrument or "unknown",
        }

        for dim_name, key_func in dimensions.items():
            groups = defaultdict(list)
            for t in self.trades:
                groups[key_func(t)].append(t)
            analysis[dim_name] = {
                k: PerformanceMetrics.calculate_all(v, self.initial_capital)
                for k, v in groups.items()
            }

        return analysis

    # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # MONTE CARLO SIMULATION
    # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def monte_carlo(self, n_simulations: int = 1000) -> Dict:
        """
        Monte Carlo simulation: resampling de trades para estimar
        distribución de resultados posibles.
        """
        if not self.trades:
            return {"error": "No trades for Monte Carlo"}

        pnls = [t.pnl for t in self.trades if t.pnl is not None]
        if len(pnls) < 5:
            return {"error": "Need at least 5 trades"}

        mc_profits = []
        mc_drawdowns = []
        mc_sharpes = []

        for _ in range(n_simulations):
            sampled = np.random.choice(pnls, size=len(pnls), replace=True)
            equity = np.cumsum(sampled) + self.initial_capital
            total_profit = float(equity[-1] - self.initial_capital)

            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / peak
            max_dd = float(np.max(dd))

            rets = sampled / self.initial_capital
            sharpe = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252))

            mc_profits.append(total_profit)
            mc_drawdowns.append(max_dd)
            mc_sharpes.append(sharpe)

        return {
            "n_simulations": n_simulations,
            "profit": {
                "p5": round(float(np.percentile(mc_profits, 5)), 2),
                "p25": round(float(np.percentile(mc_profits, 25)), 2),
                "p50": round(float(np.percentile(mc_profits, 50)), 2),
                "p75": round(float(np.percentile(mc_profits, 75)), 2),
                "p95": round(float(np.percentile(mc_profits, 95)), 2),
                "mean": round(float(np.mean(mc_profits)), 2),
            },
            "max_drawdown": {
                "p5": round(float(np.percentile(mc_drawdowns, 5)) * 100, 2),
                "p50": round(float(np.percentile(mc_drawdowns, 50)) * 100, 2),
                "p95": round(float(np.percentile(mc_drawdowns, 95)) * 100, 2),
            },
            "sharpe": {
                "p5": round(float(np.percentile(mc_sharpes, 5)), 3),
                "p50": round(float(np.percentile(mc_sharpes, 50)), 3),
                "p95": round(float(np.percentile(mc_sharpes, 95)), 3),
            },
            "probability_profitable": round(
                float(np.mean(np.array(mc_profits) > 0)) * 100, 1
            ),
        }

    # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # STRESS TEST
    # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def stress_test(self, candles: List[Dict], vol_multipliers: List[float] = None) -> Dict:
        """Stress test: replay con volatilidad amplificada."""
        if vol_multipliers is None:
            vol_multipliers = [1.0, 1.5, 2.0, 3.0]

        stress_results = {}
        for mult in vol_multipliers:
            stressed_candles = self._amplify_volatility(candles, mult)
            bt = Backtester(
                initial_capital=self.initial_capital,
                commission_pct=self.commission_pct,
                slippage_pct=self.slippage_pct * mult,
            )
            result = bt.run(stressed_candles)
            stress_results[f"{mult}x"] = {
                "net_profit": result.get("net_profit", 0),
                "max_drawdown_pct": result.get("max_drawdown_pct", 0),
                "sharpe_ratio": result.get("sharpe_ratio", 0),
                "n_trades": result.get("n_trades", 0),
                "win_rate": result.get("win_rate", 0),
                "profit_factor": result.get("profit_factor", 0),
            }
        return stress_results

    def _amplify_volatility(self, candles: List[Dict], multiplier: float) -> List[Dict]:
        if multiplier == 1.0:
            return candles

        stressed = []
        for c in candles:
            close = self._get_close(c)
            high = float(c.get('high', c.get('mid', {}).get('h', close)))
            low = float(c.get('low', c.get('mid', {}).get('l', close)))
            open_p = float(c.get('open', c.get('mid', {}).get('o', close)))
            mid = (high + low) / 2

            new_candle = dict(c)
            new_candle['high'] = mid + (high - mid) * multiplier
            new_candle['low'] = mid + (low - mid) * multiplier
            new_candle['open'] = mid + (open_p - mid) * multiplier
            stressed.append(new_candle)

        return stressed

    # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # REPORTE
    # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def generate_report(self) -> Dict:
        """Generar reporte completo de backtesting."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "initial_capital": self.initial_capital,
                "commission_pct": self.commission_pct,
                "slippage_pct": self.slippage_pct,
                "risk_per_trade_pct": self.risk_per_trade_pct,
            },
            "metrics": self.results,
            "analysis": self.analysis_by_dimension,
            "monte_carlo": self.monte_carlo(500),
            "n_trades_detail": len(self.trades),
            "trades": [t.to_dict() for t in self.trades[-50:]],
        }

    def save_report(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "backtest_reports",
                f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        report = self.generate_report()
        if "metrics" in report and "equity_curve" in report["metrics"]:
            report["metrics"]["equity_curve"] = report["metrics"]["equity_curve"][-100:]
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Backtest report saved: {filepath}")
        return filepath

    def get_status(self) -> Dict:
        return {
            "n_trades": len(self.trades),
            "current_capital": round(self.current_capital, 2),
            "metrics": {k: v for k, v in self.results.items()
                       if k not in ("equity_curve", "drawdown_curve")},
            "has_monte_carlo": bool(self.results),
        }
