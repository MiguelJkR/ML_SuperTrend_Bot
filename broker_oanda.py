"""
ML SuperTrend v51 - OANDA Broker Adapter
=========================================

Wraps existing OandaClient to conform to BrokerClient interface.
This is a thin adapter - all real logic stays in oanda_client.py.
"""

import logging
from typing import Optional, List, Dict
from broker_base import BrokerClient, BrokerOrderResult, BrokerTrade

logger = logging.getLogger(__name__)


class OandaBroker(BrokerClient):
    """OANDA broker adapter implementing BrokerClient interface."""
    
    def __init__(self, config: dict):
        from oanda_client import OandaClient
        self._client = OandaClient(config)
    
    @property
    def broker_name(self) -> str:
        return "OANDA"
    
    @property
    def client(self):
        """Direct access to underlying OandaClient for backward compatibility."""
        return self._client
    
    def test_connection(self) -> bool:
        return self._client.test_connection()
    
    def get_account_balance(self) -> float:
        result = self._client.get_account_summary()
        if result and "account" in result:
            return float(result["account"].get("balance", 0))
        return 0.0
    
    def get_candles(self, instrument: str, granularity: str,
                    count: int = 300, from_time: str = None) -> Optional[List[Dict]]:
        return self._client.get_candles(instrument, granularity, count, from_time)
    
    def get_current_price(self, instrument: str) -> Optional[Dict]:
        return self._client.get_current_price(instrument)
    
    def market_order(self, instrument: str, units: float,
                     sl_price: float = None, tp_price: float = None) -> Optional[BrokerOrderResult]:
        result = self._client.market_order(instrument, int(units), sl_price, tp_price)
        if result and "orderFillTransaction" in result:
            fill = result["orderFillTransaction"]
            return BrokerOrderResult(
                success=True,
                order_id=str(fill.get("id", "")),
                fill_price=float(fill.get("price", 0)),
                units=float(fill.get("units", 0)),
                raw=result,
            )
        return BrokerOrderResult(success=False, message=str(result))
    
    def modify_trade_sl(self, trade_id: str, sl_price: float) -> bool:
        result = self._client.modify_trade_sl(trade_id, sl_price)
        return result is not None
    
    def close_trade(self, trade_id: str, units: str = "ALL") -> bool:
        result = self._client.close_trade(trade_id, units)
        return result is not None
    
    def get_open_trades(self) -> Optional[List[BrokerTrade]]:
        trades = self._client.get_open_trades()
        if trades is None:
            return None
        result = []
        for t in trades:
            units = float(t.get("currentUnits", 0))
            result.append(BrokerTrade(
                trade_id=t["id"],
                instrument=t["instrument"],
                direction="LONG" if units > 0 else "SHORT",
                units=abs(units),
                entry_price=float(t.get("price", 0)),
                stop_loss=float(t.get("stopLossOrder", {}).get("price", 0)),
                take_profit=float(t.get("takeProfitOrder", {}).get("price", 0)),
                unrealized_pnl=float(t.get("unrealizedPL", 0)),
                open_time=t.get("openTime", ""),
            ))
        return result
