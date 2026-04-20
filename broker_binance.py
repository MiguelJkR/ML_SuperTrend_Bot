"""
ML SuperTrend v51 - Binance Futures Broker Client
===================================================

Implements BrokerClient for Binance USDT-M Futures.
Supports: BTCUSDT, ETHUSDT, and other USDT-margined perpetual contracts.

Requirements:
    pip install python-binance

Configuration:
    BINANCE_CONFIG = {
        "api_key": "your_api_key",
        "api_secret": "your_api_secret",
        "testnet": True,  # Use testnet for demo
    }
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict

from broker_base import BrokerClient, BrokerOrderResult, BrokerTrade

logger = logging.getLogger(__name__)

# Timeframe mapping: our format -> Binance format
GRANULARITY_MAP = {
    "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
    "H1": "1h", "H4": "4h", "D": "1d", "W": "1w",
}

try:
    from binance.client import Client as BinanceSDK
    from binance.enums import (
        SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET,
        FUTURE_ORDER_TYPE_STOP_MARKET, FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
    )
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    logger.warning("python-binance not installed. Run: pip install python-binance")


class BinanceBroker(BrokerClient):
    """
    Binance USDT-M Futures broker client.
    
    Supports spot-like order flow with SL/TP as separate stop orders.
    Uses Binance Futures Testnet when testnet=True.
    """
    
    def __init__(self, config: dict):
        if not BINANCE_AVAILABLE:
            raise ImportError("python-binance is required. Install: pip install python-binance")
        
        self.api_key = config["api_key"]
        self.api_secret = config["api_secret"]
        self.testnet = config.get("testnet", True)
        
        self._client = BinanceSDK(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet,
        )
        
        # Cache for symbol info (precision, min qty, etc.)
        self._symbol_info_cache = {}
        logger.info(f"Binance {'Testnet' if self.testnet else 'LIVE'} client initialized")
    
    @property
    def broker_name(self) -> str:
        return "BINANCE"
    
    def normalize_symbol(self, symbol: str) -> str:
        """Convert unified 'BTC_USDT' to Binance 'BTCUSDT'."""
        return symbol.replace("_", "")
    
    def denormalize_symbol(self, broker_symbol: str) -> str:
        """Convert 'BTCUSDT' back to 'BTC_USDT'."""
        # Common USDT pairs
        if broker_symbol.endswith("USDT"):
            base = broker_symbol[:-4]
            return f"{base}_USDT"
        return broker_symbol
    
    def _get_symbol_info(self, symbol: str) -> dict:
        """Get symbol precision and filters from exchange info."""
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]
        try:
            info = self._client.futures_exchange_info()
            for s in info.get("symbols", []):
                if s["symbol"] == symbol:
                    # Extract precision
                    price_precision = s.get("pricePrecision", 2)
                    qty_precision = s.get("quantityPrecision", 3)
                    min_qty = 0.001
                    for f in s.get("filters", []):
                        if f["filterType"] == "LOT_SIZE":
                            min_qty = float(f.get("minQty", 0.001))
                    
                    result = {
                        "price_precision": price_precision,
                        "qty_precision": qty_precision,
                        "min_qty": min_qty,
                    }
                    self._symbol_info_cache[symbol] = result
                    return result
        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
        return {"price_precision": 2, "qty_precision": 3, "min_qty": 0.001}
    
    def test_connection(self) -> bool:
        try:
            balance = self._client.futures_account_balance()
            if balance:
                usdt = next((b for b in balance if b["asset"] == "USDT"), None)
                if usdt:
                    logger.info(f"Binance connected. USDT Balance: {usdt['balance']}")
                    return True
        except Exception as e:
            logger.error(f"Binance connection test failed: {e}")
        return False
    
    def get_account_balance(self) -> float:
        try:
            balance = self._client.futures_account_balance()
            usdt = next((b for b in balance if b["asset"] == "USDT"), None)
            return float(usdt["balance"]) if usdt else 0.0
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0
    
    def get_candles(self, instrument: str, granularity: str,
                    count: int = 300, from_time: str = None) -> Optional[List[Dict]]:
        """Fetch Binance futures klines."""
        symbol = self.normalize_symbol(instrument)
        interval = GRANULARITY_MAP.get(granularity, "1h")
        
        try:
            kwargs = {"symbol": symbol, "interval": interval, "limit": min(count, 1500)}
            
            if from_time:
                # Parse ISO to ms timestamp
                try:
                    dt = datetime.fromisoformat(from_time.replace("Z", "+00:00"))
                    kwargs["startTime"] = int(dt.timestamp() * 1000)
                except ValueError:
                    pass
            
            klines = self._client.futures_klines(**kwargs)
            
            candles = []
            for k in klines:
                # Binance kline format: [open_time, o, h, l, c, volume, close_time, ...]
                open_time_ms = k[0]
                close_time_ms = k[6]
                is_complete = close_time_ms < int(time.time() * 1000)
                
                candles.append({
                    "time": datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).isoformat(),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "complete": is_complete,
                })
            
            return candles
            
        except Exception as e:
            logger.error(f"Failed to get Binance candles for {symbol}: {e}")
            return None
    
    def get_current_price(self, instrument: str) -> Optional[Dict]:
        """Get current price from Binance order book."""
        symbol = self.normalize_symbol(instrument)
        try:
            ticker = self._client.futures_orderbook_ticker(symbol=symbol)
            bid = float(ticker["bidPrice"])
            ask = float(ticker["askPrice"])
            return {
                "bid": bid, "ask": ask,
                "mid": (bid + ask) / 2,
                "spread": ask - bid,
                "time": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None
    
    def market_order(self, instrument: str, units: float,
                     sl_price: float = None, tp_price: float = None) -> Optional[BrokerOrderResult]:
        """
        Place Binance futures market order with optional SL/TP.
        
        Binance requires SL/TP as separate stop orders (not attached to the main order).
        """
        symbol = self.normalize_symbol(instrument)
        info = self._get_symbol_info(symbol)
        
        side = SIDE_BUY if units > 0 else SIDE_SELL
        qty = round(abs(units), info["qty_precision"])
        
        if qty < info["min_qty"]:
            return BrokerOrderResult(success=False, message=f"Qty {qty} below minimum {info['min_qty']}")
        
        try:
            # Main market order
            result = self._client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=qty,
            )
            
            fill_price = float(result.get("avgPrice", 0))
            if fill_price == 0 and result.get("fills"):
                fill_price = float(result["fills"][0]["price"])
            
            order_id = str(result.get("orderId", ""))
            
            # Place SL order
            if sl_price:
                sl_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
                try:
                    self._client.futures_create_order(
                        symbol=symbol,
                        side=sl_side,
                        type=FUTURE_ORDER_TYPE_STOP_MARKET,
                        stopPrice=round(sl_price, info["price_precision"]),
                        closePosition=True,
                    )
                except Exception as e:
                    logger.warning(f"Failed to place SL order: {e}")
            
            # Place TP order
            if tp_price:
                tp_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
                try:
                    self._client.futures_create_order(
                        symbol=symbol,
                        side=tp_side,
                        type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                        stopPrice=round(tp_price, info["price_precision"]),
                        closePosition=True,
                    )
                except Exception as e:
                    logger.warning(f"Failed to place TP order: {e}")
            
            logger.info(f"Binance order filled: {symbol} {side} {qty} @ {fill_price}")
            
            return BrokerOrderResult(
                success=True,
                order_id=order_id,
                fill_price=fill_price,
                units=qty if side == SIDE_BUY else -qty,
                raw=result,
            )
            
        except Exception as e:
            logger.error(f"Binance market order failed: {e}")
            return BrokerOrderResult(success=False, message=str(e))
    
    def modify_trade_sl(self, trade_id: str, sl_price: float) -> bool:
        """
        Modify SL on Binance. Requires canceling old stop order and placing new one.
        trade_id here is the symbol (Binance doesn't have trade-level SL).
        """
        symbol = trade_id  # For Binance, pass symbol as trade_id
        info = self._get_symbol_info(symbol)
        
        try:
            # Cancel existing stop orders
            orders = self._client.futures_get_open_orders(symbol=symbol)
            for o in orders:
                if o["type"] in ("STOP_MARKET", "STOP"):
                    self._client.futures_cancel_order(symbol=symbol, orderId=o["orderId"])
            
            # Get current position to determine SL side
            positions = self._client.futures_position_information(symbol=symbol)
            for p in positions:
                amt = float(p.get("positionAmt", 0))
                if amt != 0:
                    sl_side = SIDE_SELL if amt > 0 else SIDE_BUY
                    self._client.futures_create_order(
                        symbol=symbol,
                        side=sl_side,
                        type=FUTURE_ORDER_TYPE_STOP_MARKET,
                        stopPrice=round(sl_price, info["price_precision"]),
                        closePosition=True,
                    )
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to modify SL on Binance: {e}")
            return False
    
    def close_trade(self, trade_id: str, units: str = "ALL") -> bool:
        """Close position on Binance. trade_id = symbol."""
        symbol = trade_id
        try:
            positions = self._client.futures_position_information(symbol=symbol)
            for p in positions:
                amt = float(p.get("positionAmt", 0))
                if amt != 0:
                    side = SIDE_SELL if amt > 0 else SIDE_BUY
                    qty = abs(amt)
                    self._client.futures_create_order(
                        symbol=symbol, side=side,
                        type=ORDER_TYPE_MARKET, quantity=qty,
                    )
                    # Cancel remaining stop orders
                    self._client.futures_cancel_all_open_orders(symbol=symbol)
                    logger.info(f"Closed Binance position: {symbol} {qty}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to close Binance position: {e}")
            return False
    
    def get_open_trades(self) -> Optional[List[BrokerTrade]]:
        """Get all open Binance futures positions."""
        try:
            positions = self._client.futures_position_information()
            result = []
            for p in positions:
                amt = float(p.get("positionAmt", 0))
                if amt == 0:
                    continue
                symbol = p["symbol"]
                result.append(BrokerTrade(
                    trade_id=symbol,
                    instrument=self.denormalize_symbol(symbol),
                    direction="LONG" if amt > 0 else "SHORT",
                    units=abs(amt),
                    entry_price=float(p.get("entryPrice", 0)),
                    stop_loss=0,  # Binance doesn't attach SL to position
                    take_profit=0,
                    unrealized_pnl=float(p.get("unrealizedProfit", 0)),
                    open_time=str(p.get("updateTime", "")),
                ))
            return result
        except Exception as e:
            logger.error(f"Failed to get Binance positions: {e}")
            return None
