"""
ML SuperTrend v51 Trading Bot - Paper Trading Module
Simulates trades using real market data without executing on OANDA.
Tracks PnL, win/loss, and all risk management as if live.
"""

import logging
import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from config import InstrumentConfig

logger = logging.getLogger(__name__)

PAPER_LOG_FILE = "paper_trades.json"


@dataclass
class PaperTrade:
    """A simulated trade."""
    trade_id: str
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    current_sl: float
    units: int
    atr_at_entry: float
    strength: float
    reasons: List[str]
    open_time: str
    close_time: str = ""
    close_price: float = 0.0
    pnl_usd: float = 0.0
    pnl_pips: float = 0.0
    status: str = "OPEN"  # "OPEN", "WIN", "LOSS", "BE"
    be_triggered: bool = False
    trailing_active: bool = False
    highest_price: float = 0.0
    lowest_price: float = 999999.0
    sl_moves: List[str] = field(default_factory=list)


class PaperTrader:
    """
    Paper trading engine - simulates execution using real market prices.
    Runs in parallel with live trading for strategy validation.
    """

    def __init__(self, initial_balance: float = 100000.0):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.equity = initial_balance
        self.open_trades: Dict[str, PaperTrade] = {}
        self.closed_trades: List[PaperTrade] = []
        self.trade_counter = 0
        self.stats = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "breakevens": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "peak_balance": initial_balance,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }
        self._load_history()
        logger.info(f"PaperTrader initialized: balance=${initial_balance:,.2f}")

    def _next_id(self) -> str:
        self.trade_counter += 1
        return f"PAPER-{self.trade_counter:04d}"

    def open_trade(self, instrument: InstrumentConfig, direction: str,
                   entry_price: float, sl_price: float, atr_value: float,
                   units: int, strength: float, reasons: List[str]) -> PaperTrade:
        """Simulate opening a trade."""
        trade_id = self._next_id()
        trade = PaperTrade(
            trade_id=trade_id,
            symbol=instrument.symbol,
            direction=direction,
            entry_price=entry_price,
            current_sl=sl_price,
            units=abs(units),
            atr_at_entry=atr_value,
            strength=strength,
            reasons=reasons,
            open_time=datetime.now(timezone.utc).isoformat(),
            highest_price=entry_price,
            lowest_price=entry_price,
        )
        self.open_trades[trade_id] = trade
        self.stats["total_trades"] += 1

        logger.info(
            f"[PAPER] OPENED: {trade_id} {direction} {instrument.symbol} "
            f"@ {entry_price:.5f} | SL={sl_price:.5f} | Units={units}"
        )
        return trade

    def update_price(self, symbol: str, current_price: float, instrument: InstrumentConfig):
        """Update all open trades for a symbol with current price."""
        for trade_id, trade in list(self.open_trades.items()):
            if trade.symbol != symbol:
                continue

            # Track highest/lowest
            trade.highest_price = max(trade.highest_price, current_price)
            trade.lowest_price = min(trade.lowest_price, current_price)

            # Check if SL hit
            if trade.direction == "LONG" and current_price <= trade.current_sl:
                self._close_trade(trade, trade.current_sl, "SL_HIT")
                continue
            elif trade.direction == "SHORT" and current_price >= trade.current_sl:
                self._close_trade(trade, trade.current_sl, "SL_HIT")
                continue

            # === BREAKEVEN ===
            if not trade.be_triggered:
                be_threshold = instrument.auto_be_tr * trade.atr_at_entry
                if trade.direction == "LONG" and current_price >= trade.entry_price + be_threshold:
                    new_sl = trade.entry_price + instrument.be_offset
                    trade.current_sl = new_sl
                    trade.be_triggered = True
                    trade.sl_moves.append(f"BE -> {new_sl:.5f}")
                    logger.info(f"[PAPER] BE {trade_id}: SL -> {new_sl:.5f}")
                elif trade.direction == "SHORT" and current_price <= trade.entry_price - be_threshold:
                    new_sl = trade.entry_price - instrument.be_offset
                    trade.current_sl = new_sl
                    trade.be_triggered = True
                    trade.sl_moves.append(f"BE -> {new_sl:.5f}")
                    logger.info(f"[PAPER] BE {trade_id}: SL -> {new_sl:.5f}")

            # === TRAILING ===
            if trade.be_triggered and not trade.trailing_active:
                if trade.direction == "LONG":
                    pnl_atr = (current_price - trade.entry_price) / trade.atr_at_entry
                else:
                    pnl_atr = (trade.entry_price - current_price) / trade.atr_at_entry
                if pnl_atr >= instrument.auto_trail_tr:
                    trade.trailing_active = True
                    logger.info(f"[PAPER] Trailing ON {trade_id}")

            if trade.trailing_active:
                trail_offset = instrument.auto_trail_m * trade.atr_at_entry
                if trade.direction == "LONG":
                    new_sl = current_price - trail_offset
                    if new_sl > trade.current_sl:
                        trade.sl_moves.append(f"TRAIL -> {new_sl:.5f}")
                        trade.current_sl = new_sl
                elif trade.direction == "SHORT":
                    new_sl = current_price + trail_offset
                    if new_sl < trade.current_sl:
                        trade.sl_moves.append(f"TRAIL -> {new_sl:.5f}")
                        trade.current_sl = new_sl

        # Update equity
        self._update_equity(current_price)

    def _close_trade(self, trade: PaperTrade, close_price: float, reason: str):
        """Close a paper trade and record results."""
        trade.close_price = close_price
        trade.close_time = datetime.now(timezone.utc).isoformat()

        # Calculate PnL
        if trade.direction == "LONG":
            pnl_raw = (close_price - trade.entry_price) * trade.units
            pnl_pips = (close_price - trade.entry_price)
        else:
            pnl_raw = (trade.entry_price - close_price) * trade.units
            pnl_pips = (trade.entry_price - close_price)

        # Adjust for XAU (1 pip = $1) vs EUR (1 pip = $0.0001 * units)
        if "XAU" not in trade.symbol:
            pnl_pips *= 10000  # Convert to pips

        trade.pnl_usd = pnl_raw
        trade.pnl_pips = pnl_pips

        # Classify result
        if pnl_raw > 0:
            trade.status = "WIN"
            self.stats["wins"] += 1
        elif pnl_raw < -0.01:
            trade.status = "LOSS"
            self.stats["losses"] += 1
        else:
            trade.status = "BE"
            self.stats["breakevens"] += 1

        self.stats["total_pnl"] += pnl_raw
        self.stats["best_trade"] = max(self.stats["best_trade"], pnl_raw)
        self.stats["worst_trade"] = min(self.stats["worst_trade"], pnl_raw)

        # Update balance
        self.balance += pnl_raw
        if self.balance > self.stats["peak_balance"]:
            self.stats["peak_balance"] = self.balance
        drawdown = self.stats["peak_balance"] - self.balance
        if drawdown > self.stats["max_drawdown"]:
            self.stats["max_drawdown"] = drawdown

        # Move to closed
        del self.open_trades[trade.trade_id]
        self.closed_trades.append(trade)

        logger.info(
            f"[PAPER] CLOSED: {trade.trade_id} {trade.direction} {trade.symbol} | "
            f"Result={trade.status} | PnL=${pnl_raw:+.2f} | "
            f"Entry={trade.entry_price:.5f} Close={close_price:.5f} | {reason}"
        )

        self._save_history()

    def _update_equity(self, current_price: float):
        """Recalculate equity based on open trades."""
        unrealized = 0.0
        for trade in self.open_trades.values():
            if trade.direction == "LONG":
                unrealized += (current_price - trade.entry_price) * trade.units
            else:
                unrealized += (trade.entry_price - current_price) * trade.units
        self.equity = self.balance + unrealized

    def get_status(self) -> Dict:
        """Get paper trading status for dashboard."""
        total = self.stats["wins"] + self.stats["losses"] + self.stats["breakevens"]
        win_rate = (self.stats["wins"] / total * 100) if total > 0 else 0

        return {
            "mode": "PAPER",
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "initial_balance": self.initial_balance,
            "return_pct": round((self.balance - self.initial_balance) / self.initial_balance * 100, 2),
            "total_pnl": round(self.stats["total_pnl"], 2),
            "open_trades": [
                {
                    "id": t.trade_id, "symbol": t.symbol, "direction": t.direction,
                    "entry": t.entry_price, "sl": t.current_sl, "units": t.units,
                    "be": t.be_triggered, "trailing": t.trailing_active,
                    "pnl_atr": round((t.highest_price - t.entry_price) / t.atr_at_entry, 2)
                    if t.direction == "LONG" else
                    round((t.entry_price - t.lowest_price) / t.atr_at_entry, 2),
                }
                for t in self.open_trades.values()
            ],
            "stats": {
                "total": total,
                "wins": self.stats["wins"],
                "losses": self.stats["losses"],
                "breakevens": self.stats["breakevens"],
                "win_rate": round(win_rate, 1),
                "best_trade": round(self.stats["best_trade"], 2),
                "worst_trade": round(self.stats["worst_trade"], 2),
                "max_drawdown": round(self.stats["max_drawdown"], 2),
            },
            "recent_closed": [
                {
                    "id": t.trade_id, "symbol": t.symbol, "direction": t.direction,
                    "result": t.status, "pnl": round(t.pnl_usd, 2),
                    "entry": t.entry_price, "close": t.close_price,
                }
                for t in self.closed_trades[-10:]  # Last 10
            ],
        }

    def _save_history(self):
        """Save closed trades to JSON file."""
        try:
            data = {
                "balance": self.balance,
                "stats": self.stats,
                "trade_counter": self.trade_counter,
                "closed_trades": [asdict(t) for t in self.closed_trades[-100:]],
            }
            with open(PAPER_LOG_FILE, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Paper history save error: {e}")

    def _load_history(self):
        """Load previous paper trading session."""
        if os.path.exists(PAPER_LOG_FILE):
            try:
                with open(PAPER_LOG_FILE) as f:
                    data = json.load(f)
                self.balance = data.get("balance", self.initial_balance)
                self.stats.update(data.get("stats", {}))
                self.trade_counter = data.get("trade_counter", 0)
                logger.info(f"Paper history loaded: balance=${self.balance:,.2f}, trades={self.stats['total_trades']}")
            except Exception as e:
                logger.warning(f"Could not load paper history: {e}")
