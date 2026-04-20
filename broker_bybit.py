"""
ML SuperTrend v51 - Bybit Broker Client
=========================================

Implements BrokerClient for Bybit V5 API (USDT Perpetuals).
Supports: BTCUSDT, ETHUSDT, and other linear perpetual contracts.

Requirements:
    pip install pybit

Configuration:
    BYBIT_CONFIG = {
        "api_key": "your_api_key",
        "api_secret": "your_api_secret",
        "testnet": True,
    }
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict

from broker_base import BrokerClient, BrokerOrderResult, BrokerTrade

logger = logging.getLogger(__name__)

# Timeframe mapping: our format -> Bybit interval
GRANULARITY_MAP = {
    "M1": "1", "M5": "5", "M15": "15", "M30": "30",
    "H1": "60", "H4": "240", "D": "D", "W": "W",
}

try:
    from pybit.unified_trading import HTTP as BybitHTTP
    BYBIT_AVAILABLE = True
except ImportError:
    BYBIT_AVAILABLE = False
    logger.warning("pybit not installed. Run: pip install pybit")


class BybitBroker(BrokerClient):
    """
    Bybit V5 API broker client for USDT-margined perpetual futures.
    
    Uses the unified trading V5 endpoints.
    """
    
    def __init__(self, config: dict):
        if not BYBIT_AVAILABLE:
            raise ImportError("pybit is required. Install: pip install pybit")
        
        self.api_key = config["api_key"]
        self.api_secret = config["api_secret"]
        self.testnet = config.get("testnet", True)
        
        self._client = BybitHTTP(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet,
        )
        
        self._symbol_info_cache = {}
        logger.info(f"Bybit {'Testnet' if self.testnet else 'LIVE'} client initialized")
    
    @property
    def broker_name(self) -> str:
        return "BYBIT"
    
    def normalize_symbol(self, symbol: str) -> str:
        """Convert unified 'BTC_USDT' to Bybit 'BTCUSDT'."""
        return symbol.replace("_", "")
    
    def denormalize_symbol(self, broker_symbol: str) -> str:
        """Convert 'BTCUSDT' back to 'BTC_USDT'."""
        if broker_symbol.endswith("USDT"):
            base = broker_symbol[:-4]
            return f"{base}_USDT"
        return broker_symbol
    
    def _get_symbol_info(self, symbol: str) -> dict:
        """Get symbol tick size and lot size from Bybit."""
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]
        try:
            resp = self._client.get_instruments_info(category="linear", symbol=symbol)
            if resp["retCode"] == 0 and resp["result"]["list"]:
                info = resp["result"]["list"][0]
                lot_filter = info.get("lotSizeFilter", {})
                price_filter = info.get("priceFilter", {})
                
                # Count decimal places for precision
                tick_size = price_filter.get("tickSize", "0.01")
                qty_step = lot_filter.get("qtyStep", "0.001")
                
                price_prec = len(tick_size.rstrip('0').split('.')[-1]) if '.' in tick_size else 0
                qty_prec = len(qty_step.rstrip('0').split('.')[-1]) if '.' in qty_step else 0
                
                result = {
                    "price_precision": price_prec,
                    "qty_precision": qty_prec,
                    "min_qty": float(lot_filter.get("minOrderQty", 0.001)),
                    "tick_size": float(tick_size),
                }
                self._symbol_info_cache[symbol] = result
                return result
        except Exception as e:
            logger.error(f"Failed to get Bybit symbol info for {symbol}: {e}")
        return {"price_precision": 2, "qty_precision": 3, "min_qty": 0.001, "tick_size": 0.01}
    
    def test_connection(self) -> bool:
        try:
            resp = self._client.get_wallet_balance(accountType="UNIFIED")
            if resp["retCode"] == 0:
                coins = resp["result"]["list"][0]["coin"] if resp["result"]["list"] else []
                usdt = next((c for c in coins if c["coin"] == "USDT"), None)
                if usdt:
                    logger.info(f"Bybit connected. USDT Balance: {usdt['walletBalance']}")
                return True
        except Exception as e:
            logger.error(f"Bybit connection test failed: {e}")
        return False
    
    def get_account_balance(self) -> float:
        try:
            resp = self._client.get_wallet_balance(accountType="UNIFIED")
            if resp["retCode"] == 0 and resp["result"]["list"]:
                coins = resp["result"]["list"][0]["coin"]
                usdt = next((c for c in coins if c["coin"] == "USDT"), None)
                return float(usdt["walletBalance"]) if usdt else 0.0
        except Exception as e:
            logger.error(f"Failed to get Bybit balance: {e}")
        return 0.0
    
    def get_candles(self, instrument: str, granularity: str,
                    count: int = 300, from_time: str = None) -> Optional[List[Dict]]:
        """Fetch Bybit klines (V5 API)."""
        symbol = self.normalize_symbol(instrument)
        interval = GRANULARITY_MAP.get(granularity, "60")
        
        try:
            kwargs = {"category": "linear", "symbol": symbol, 
                      "interval": interval, "limit": min(count, 1000)}
            
            if from_time:
                try:
                    dt = datetime.fromisoformat(from_time.replace("Z", "+00:00"))
                    kwargs["start"] = int(dt.timestamp() * 1000)
                except ValueError:
                    pass
            
            resp = self._client.get_kline(**kwargs)
            
            if resp["retCode"] != 0:
                logger.error(f"Bybit kline error: {resp['retMsg']}")
                return None
            
            candles = []
            # Bybit returns newest first, so reverse
            klines = resp["result"]["list"][::-1]
            
            for k in klines:
                # Bybit V5 kline: [startTime, open, high, low, close, volume, turnover]
                open_time_ms = int(k[0])
                candles.append({
                    "time": datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).isoformat(),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "complete": open_time_ms < int(time.time() * 1000),
                })
            
            return candles
            
        except Exception as e:
            logger.error(f"Failed to get Bybit candles for {symbol}: {e}")
            return None
    
    def get_current_price(self, instrument: str) -> Optional[Dict]:
        """Get current price from Bybit."""
        symbol = self.normalize_symbol(instrument)
        try:
            resp = self._client.get_orderbook(category="linear", symbol=symbol, limit=1)
            if resp["retCode"] == 0:
                bids = resp["result"]["b"]
                asks = resp["result"]["a"]
                bid = float(bids[0][0]) if bids else 0
                ask = float(asks[0][0]) if asks else 0
                return {
                    "bid": bid, "ask": ask,
                    "mid": (bid + ask) / 2,
                    "spread": ask - bid,
                    "time": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as e:
            logger.error(f"Failed to get Bybit price for {symbol}: {e}")
        return None
    
    def market_order(self, instrument: str, units: float,
                     sl_price: float = None, tp_price: float = None) -> Optional[BrokerOrderResult]:
        """Place Bybit linear market order with optional SL/TP."""
        symbol = self.normalize_symbol(instrument)
        info = self._get_symbol_info(symbol)
        
        side = "Buy" if units > 0 else "Sell"
        qty = str(round(abs(units), info["qty_precision"]))
        
        if float(qty) < info["min_qty"]:
            return BrokerOrderResult(success=False, message=f"Qty {qty} below minimum {info['min_qty']}")
        
        try:
            kwargs = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                "qty": qty,
            }
            
            # Bybit V5 supports SL/TP on the order itself
            if sl_price:
                kwargs["stopLoss"] = str(round(sl_price, info["price_precision"]))
                kwargs["slTriggerBy"] = "LastPrice"
            if tp_price:
                kwargs["takeProfit"] = str(round(tp_price, info["price_precision"]))
                kwargs["tpTriggerBy"] = "LastPrice"
            
            resp = self._client.place_order(**kwargs)
            
            if resp["retCode"] != 0:
                return BrokerOrderResult(success=False, message=resp.get("retMsg", "Unknown error"))
            
            order_id = resp["result"].get("orderId", "")
            
            # Get fill price (may need to query)
            fill_price = 0.0
            try:
                time.sleep(0.5)
                exec_resp = self._client.get_executions(
                    category="linear", symbol=symbol, orderId=order_id
                )
                if exec_resp["retCode"] == 0 and exec_resp["result"]["list"]:
                    fill_price = float(exec_resp["result"]["list"][0]["execPrice"])
            except Exception:
                pass
            
            logger.info(f"Bybit order filled: {symbol} {side} {qty} @ {fill_price}")
            
            return BrokerOrderResult(
                success=True,
                order_id=order_id,
                fill_price=fill_price,
                units=float(qty) if side == "Buy" else -float(qty),
                raw=resp,
            )
            
        except Exception as e:
            logger.error(f"Bybit market order failed: {e}")
            return BrokerOrderResult(success=False, message=str(e))
    
    def modify_trade_sl(self, trade_id: str, sl_price: float) -> bool:
        """Modify SL on Bybit position. trade_id = symbol."""
        symbol = trade_id
        info = self._get_symbol_info(symbol)
        try:
            resp = self._client.set_trading_stop(
                category="linear",
                symbol=symbol,
                stopLoss=str(round(sl_price, info["price_precision"])),
                slTriggerBy="LastPrice",
                positionIdx=0,  # One-way mode
            )
            return resp["retCode"] == 0
        except Exception as e:
            logger.error(f"Failed to modify Bybit SL: {e}")
            return False
    
    def close_trade(self, trade_id: str, units: str = "ALL") -> bool:
        """Close Bybit position. trade_id = symbol."""
        symbol = trade_id
        try:
            # Get current position
            resp = self._client.get_positions(category="linear", symbol=symbol)
            if resp["retCode"] != 0:
                return False
            
            for p in resp["result"]["list"]:
                size = float(p.get("size", 0))
                if size > 0:
                    side = "Sell" if p["side"] == "Buy" else "Buy"
                    qty = str(size)
                    
                    close_resp = self._client.place_order(
                        category="linear", symbol=symbol,
                        side=side, orderType="Market",
                        qty=qty, reduceOnly=True,
                    )
                    
                    if close_resp["retCode"] == 0:
                        logger.info(f"Closed Bybit position: {symbol} {qty}")
                        return True
            return False
        except Exception as e:
            logger.error(f"Failed to close Bybit position: {e}")
            return False
    
    def get_open_trades(self) -> Optional[List[BrokerTrade]]:
        """Get all open Bybit linear positions."""
        try:
            resp = self._client.get_positions(category="linear", settleCoin="USDT")
            if resp["retCode"] != 0:
                return None
            
            result = []
            for p in resp["result"]["list"]:
                size = float(p.get("size", 0))
                if size == 0:
                    continue
                
                symbol = p["symbol"]
                result.append(BrokerTrade(
                    trade_id=symbol,
                    instrument=self.denormalize_symbol(symbol),
                    direction="LONG" if p["side"] == "Buy" else "SHORT",
                    units=size,
                    entry_price=float(p.get("avgPrice", 0)),
                    stop_loss=float(p.get("stopLoss", 0)),
                    take_profit=float(p.get("takeProfit", 0)),
                    unrealized_pnl=float(p.get("unrealisedPnl", 0)),
                    open_time=str(p.get("updatedTime", "")),
                ))
            return result
        except Exception as e:
            logger.error(f"Failed to get Bybit positions: {e}")
            return None
