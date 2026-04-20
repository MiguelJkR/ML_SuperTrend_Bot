"""
ML SuperTrend v51 - Crypto Exchange Client
=============================================
Unified client for Binance Futures and Bybit V5.
Provides the same interface as OandaClient for seamless integration.

Supports:
  - Market data (candles, prices, order book)
  - Futures trading (market orders, stop loss, take profit)
  - Account info (balance, positions, PnL)
  - Testnet mode for paper trading

Usage:
    from crypto_client import CryptoClient
    client = CryptoClient(config=BINANCE_CONFIG, exchange="BINANCE")
    candles = client.get_candles("BTCUSDT", "1h", count=300)
    client.market_order("BTCUSDT", units=0.001, sl_price=60000)
"""

import logging
import time
import hmac
import hashlib
import json
from typing import Dict, List, Optional
from datetime import datetime, timezone
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

# Try to use requests for HTTP
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not available -- crypto client won't work")


# =====================================================================
# SYMBOL MAPPING: Internal format <-> Exchange format
# =====================================================================
SYMBOL_MAP = {
    # Internal -> Binance Futures
    "BTC_USDT": "BTCUSDT",
    "ETH_USDT": "ETHUSDT",
    "SOL_USDT": "SOLUSDT",
    "XRP_USDT": "XRPUSDT",
    "BNB_USDT": "BNBUSDT",
    "DOGE_USDT": "DOGEUSDT",
    "ADA_USDT": "ADAUSDT",
    "AVAX_USDT": "AVAXUSDT",
}

# Timeframe mapping
TF_MAP_BINANCE = {
    "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
    "H1": "1h", "H4": "4h", "D": "1d", "W": "1w",
}

TF_MAP_BYBIT = {
    "M1": "1", "M5": "5", "M15": "15", "M30": "30",
    "H1": "60", "H4": "240", "D": "D", "W": "W",
}


class CryptoClient:
    """
    Unified crypto exchange client supporting Binance Futures and Bybit V5.
    Provides the same method signatures as OandaClient for drop-in usage.
    """

    def __init__(self, config: Dict, exchange: str = "BINANCE"):
        """
        Initialize crypto client.

        Args:
            config: Dict with api_key, api_secret, testnet (bool)
            exchange: "BINANCE" or "BYBIT"
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required for crypto client")

        self.api_key = config.get("api_key", "")
        self.api_secret = config.get("api_secret", "")
        self.testnet = config.get("testnet", True)
        self.exchange = exchange.upper()

        # Set base URLs
        if self.exchange == "BINANCE":
            self.base_url = "https://testnet.binancefuture.com" if self.testnet else "https://fapi.binance.com"
        elif self.exchange == "BYBIT":
            self.base_url = "https://api-testnet.bybit.com" if self.testnet else "https://api.bybit.com"
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")

        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": self.api_key,  # Binance
        })

        self.account_id = f"{exchange}_FUTURES"
        self._recv_window = 5000

        logger.info(f"CryptoClient initialized: {exchange} {'TESTNET' if self.testnet else 'LIVE'}")

    def _sign_binance(self, params: Dict) -> Dict:
        """Add timestamp and signature for Binance."""
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = self._recv_window
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _sign_bybit(self, params: Dict) -> Dict:
        """Add timestamp and signature for Bybit V5."""
        timestamp = str(int(time.time() * 1000))
        params_str = urlencode(sorted(params.items()))
        sign_str = f"{timestamp}{self.api_key}{self._recv_window}{params_str}"
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        self.session.headers.update({
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": str(self._recv_window),
        })
        return params

    def _request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Optional[Dict]:
        """Make HTTP request to exchange."""
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        try:
            if signed:
                if self.exchange == "BINANCE":
                    params = self._sign_binance(params)
                else:
                    params = self._sign_bybit(params)

            if method == "GET":
                resp = self.session.get(url, params=params, timeout=10)
            else:
                resp = self.session.post(url, params=params, timeout=10)

            resp.raise_for_status()
            return resp.json()

        except Exception as e:
            logger.error(f"Crypto API error ({self.exchange}): {e}")
            return None

    # ===== DATA METHODS (same interface as OandaClient) =====

    def get_candles(self, symbol: str, granularity: str, count: int = 300) -> Optional[List[Dict]]:
        """
        Get candlestick data.

        Args:
            symbol: Internal format (e.g., "BTC_USDT")
            granularity: Our timeframe (e.g., "H1", "M30")
            count: Number of candles

        Returns:
            List of candle dicts with keys: time, open, high, low, close, volume
        """
        exchange_symbol = SYMBOL_MAP.get(symbol, symbol.replace("_", ""))

        if self.exchange == "BINANCE":
            tf = TF_MAP_BINANCE.get(granularity, "1h")
            data = self._request("GET", "/fapi/v1/klines", {
                "symbol": exchange_symbol,
                "interval": tf,
                "limit": count,
            })
            if not data:
                return None

            candles = []
            for k in data:
                candles.append({
                    "time": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).isoformat(),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
            return candles

        elif self.exchange == "BYBIT":
            tf = TF_MAP_BYBIT.get(granularity, "60")
            data = self._request("GET", "/v5/market/kline", {
                "category": "linear",
                "symbol": exchange_symbol,
                "interval": tf,
                "limit": count,
            })
            if not data or "result" not in data:
                return None

            candles = []
            for k in data["result"].get("list", []):
                candles.append({
                    "time": datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc).isoformat(),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
            candles.reverse()  # Bybit returns newest first
            return candles

        return None

    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current bid/ask/mid price."""
        exchange_symbol = SYMBOL_MAP.get(symbol, symbol.replace("_", ""))

        if self.exchange == "BINANCE":
            data = self._request("GET", "/fapi/v1/ticker/bookTicker", {"symbol": exchange_symbol})
            if data:
                bid = float(data.get("bidPrice", 0))
                ask = float(data.get("askPrice", 0))
                return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2}

        elif self.exchange == "BYBIT":
            data = self._request("GET", "/v5/market/tickers", {
                "category": "linear", "symbol": exchange_symbol
            })
            if data and "result" in data:
                tickers = data["result"].get("list", [])
                if tickers:
                    t = tickers[0]
                    bid = float(t.get("bid1Price", 0))
                    ask = float(t.get("ask1Price", 0))
                    return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2}

        return None

    def get_account_summary(self) -> Optional[Dict]:
        """Get account balance and info."""
        if self.exchange == "BINANCE":
            data = self._request("GET", "/fapi/v2/account", signed=True)
            if data:
                balance = float(data.get("totalWalletBalance", 0))
                unrealized = float(data.get("totalUnrealizedProfit", 0))
                return {
                    "account": {
                        "balance": str(balance),
                        "unrealizedPL": str(unrealized),
                        "NAV": str(balance + unrealized),
                        "currency": "USDT",
                    }
                }

        elif self.exchange == "BYBIT":
            data = self._request("GET", "/v5/account/wallet-balance", {
                "accountType": "UNIFIED"
            }, signed=True)
            if data and "result" in data:
                coins = data["result"].get("list", [{}])[0].get("coin", [])
                usdt = next((c for c in coins if c["coin"] == "USDT"), {})
                balance = float(usdt.get("walletBalance", 0))
                unrealized = float(usdt.get("unrealisedPnl", 0))
                return {
                    "account": {
                        "balance": str(balance),
                        "unrealizedPL": str(unrealized),
                        "NAV": str(balance + unrealized),
                        "currency": "USDT",
                    }
                }

        return None

    def market_order(
        self,
        instrument: str,
        units: int,
        sl_price: float = None,
        tp_price: float = None,
    ) -> Optional[Dict]:
        """
        Place a market order with optional SL/TP.

        Args:
            instrument: Internal format (e.g., "BTC_USDT")
            units: Positive = LONG, Negative = SHORT
            sl_price: Stop loss price
            tp_price: Take profit price
        """
        exchange_symbol = SYMBOL_MAP.get(instrument, instrument.replace("_", ""))
        side = "BUY" if units > 0 else "SELL"
        qty = abs(units)

        if self.exchange == "BINANCE":
            params = {
                "symbol": exchange_symbol,
                "side": side,
                "type": "MARKET",
                "quantity": str(qty),
            }
            result = self._request("POST", "/fapi/v1/order", params, signed=True)

            if result and "orderId" in result:
                # Place SL/TP as separate orders
                if sl_price:
                    sl_side = "SELL" if side == "BUY" else "BUY"
                    self._request("POST", "/fapi/v1/order", {
                        "symbol": exchange_symbol,
                        "side": sl_side,
                        "type": "STOP_MARKET",
                        "stopPrice": str(sl_price),
                        "closePosition": "true",
                    }, signed=True)

                if tp_price:
                    tp_side = "SELL" if side == "BUY" else "BUY"
                    self._request("POST", "/fapi/v1/order", {
                        "symbol": exchange_symbol,
                        "side": tp_side,
                        "type": "TAKE_PROFIT_MARKET",
                        "stopPrice": str(tp_price),
                        "closePosition": "true",
                    }, signed=True)

                return {
                    "orderFillTransaction": {
                        "tradeOpened": {"tradeID": str(result["orderId"])},
                        "price": str(result.get("avgPrice", result.get("price", 0))),
                        "units": str(units),
                    }
                }

        elif self.exchange == "BYBIT":
            params = {
                "category": "linear",
                "symbol": exchange_symbol,
                "side": "Buy" if units > 0 else "Sell",
                "orderType": "Market",
                "qty": str(qty),
            }
            if sl_price:
                params["stopLoss"] = str(sl_price)
            if tp_price:
                params["takeProfit"] = str(tp_price)

            result = self._request("POST", "/v5/order/create", params, signed=True)

            if result and result.get("retCode") == 0:
                order_id = result.get("result", {}).get("orderId", "unknown")
                return {
                    "orderFillTransaction": {
                        "tradeOpened": {"tradeID": order_id},
                        "price": "0",  # Bybit doesn't return fill price immediately
                        "units": str(units),
                    }
                }

        return None

    def get_open_trades(self) -> Optional[List[Dict]]:
        """Get list of open positions."""
        if self.exchange == "BINANCE":
            data = self._request("GET", "/fapi/v2/positionRisk", signed=True)
            if data:
                positions = []
                for p in data:
                    amt = float(p.get("positionAmt", 0))
                    if amt != 0:
                        positions.append({
                            "id": p.get("symbol", ""),
                            "instrument": p.get("symbol", ""),
                            "currentUnits": str(int(amt)),
                            "unrealizedPL": p.get("unRealizedProfit", "0"),
                            "price": p.get("entryPrice", "0"),
                        })
                return positions

        elif self.exchange == "BYBIT":
            data = self._request("GET", "/v5/position/list", {
                "category": "linear", "settleCoin": "USDT"
            }, signed=True)
            if data and "result" in data:
                positions = []
                for p in data["result"].get("list", []):
                    size = float(p.get("size", 0))
                    if size > 0:
                        positions.append({
                            "id": p.get("symbol", ""),
                            "instrument": p.get("symbol", ""),
                            "currentUnits": str(int(size)),
                            "unrealizedPL": p.get("unrealisedPnl", "0"),
                            "price": p.get("avgPrice", "0"),
                        })
                return positions

        return None

    def close_trade(self, trade_id: str, units: int = None) -> Optional[Dict]:
        """Close a position."""
        # For crypto, trade_id is the symbol
        symbol = trade_id
        if self.exchange == "BINANCE":
            # Get current position to determine direction
            positions = self.get_open_trades()
            if positions:
                for p in positions:
                    if p["instrument"] == symbol:
                        current_units = int(p["currentUnits"])
                        close_side = "SELL" if current_units > 0 else "BUY"
                        close_qty = abs(units) if units else abs(current_units)
                        return self._request("POST", "/fapi/v1/order", {
                            "symbol": symbol,
                            "side": close_side,
                            "type": "MARKET",
                            "quantity": str(close_qty),
                            "reduceOnly": "true",
                        }, signed=True)
        return None
