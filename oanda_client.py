import requests
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict
from config import OANDA_LIVE

logger = logging.getLogger(__name__)


class OandaClient:
    """
    OANDA v20 REST API client for trading operations.
    Handles authentication, candle data, pricing, orders, trades, and positions.
    """

    def __init__(self, config: dict = OANDA_LIVE):
        self.token = config["token"]
        self.base_url = config["url"]
        self.stream_url = config["stream_url"]
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339"
        })

        if config["account_id"] == "AUTO":
            self.account_id = self._discover_account_id()
        else:
            self.account_id = config["account_id"]

    def _discover_account_id(self) -> str:
        try:
            url = f"{self.base_url}/v3/accounts"
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 200:
                accounts = resp.json().get("accounts", [])
                if accounts:
                    acct_id = accounts[0]["id"]
                    logger.info(f"Auto-discovered account ID: {acct_id}")
                    return acct_id
            logger.error(f"Account discovery failed: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.error(f"Account discovery error: {e}")
        return "UNKNOWN"

    def _get(self, path: str, params=None):
        url = f"{self.base_url}/v3/{path}"
        resp = self.session.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            logger.error(f"GET {path} -> {resp.status_code}: {resp.text[:200]}")
            return None
        return resp.json()

    def _post(self, path: str, data: dict):
        url = f"{self.base_url}/v3/{path}"
        resp = self.session.post(url, json=data, timeout=30)
        if resp.status_code not in (200, 201):
            logger.error(f"POST {path} -> {resp.status_code}: {resp.text[:200]}")
            return None
        return resp.json()

    def _put(self, path: str, data: dict):
        url = f"{self.base_url}/v3/{path}"
        resp = self.session.put(url, json=data, timeout=30)
        if resp.status_code not in (200, 201):
            logger.error(f"PUT {path} -> {resp.status_code}: {resp.text}")
            return None
        return resp.json()

    def get_account(self):
        return self._get(f"accounts/{self.account_id}")

    def get_account_summary(self):
        return self._get(f"accounts/{self.account_id}/summary")

    def get_candles(self, instrument, granularity, count=300, from_time=None, to_time=None):
        params = {"granularity": granularity, "price": "MBA"}
        if from_time:
            params["from"] = from_time
            params["count"] = min(count, 5000)
        else:
            params["count"] = count
        if to_time:
            params["to"] = to_time
        data = self._get(f"instruments/{instrument}/candles", params=params)
        if not data or "candles" not in data: return None
        candles = []
        for c in data["candles"]:
            if not c.get("complete", False) and c != data["candles"][-1]: continue
            mid = c["mid"]
            candles.append({"time": c["time"], "open": float(mid["o"]), "high": float(mid["h"]), "low": float(mid["l"]), "close": float(mid["c"]), "volume": int(c.get("volume", 0)), "complete": c.get("complete", False)})
        return candles

    def get_current_price(self, instrument):
        data = self._get(f"accounts/{self.account_id}/pricing", params={"instruments": instrument})
        if not data or "prices" not in data: return None
        for p in data["prices"]:
            if p["instrument"] == instrument:
                bid = float(p["bids"][0]["price"])
                ask = float(p["asks"][0]["price"])
                return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2, "spread": ask - bid, "time": p.get("time", "")}
        return None

    def _format_price(self, instrument: str, price: float) -> str:
        """Format price to correct precision for instrument."""
        if 'JPY' in instrument or price > 50:
            return f"{price:.3f}"
        return f"{price:.5f}"

    def market_order(self, instrument, units, sl_price=None, tp_price=None):
        order = {"type": "MARKET", "instrument": instrument, "units": str(units), "timeInForce": "FOK", "positionFill": "DEFAULT"}
        if sl_price: order["stopLossOnFill"] = {"price": self._format_price(instrument, sl_price)}
        if tp_price: order["takeProfitOnFill"] = {"price": self._format_price(instrument, tp_price)}
        result = self._post(f"accounts/{self.account_id}/orders", {"order": order})
        if result and "orderFillTransaction" in result:
            fill = result["orderFillTransaction"]
            logger.info(f"ORDER FILLED: {instrument} {units} units @ {fill.get('price', 'N/A')}")
            return result
        logger.warning(f"Order may not have filled: {result}")
        return result

    def modify_trade_sl(self, trade_id, sl_price, instrument=""):
        price_str = self._format_price(instrument, sl_price)
        return self._put(f"accounts/{self.account_id}/trades/{trade_id}/orders", {"stopLoss": {"price": price_str, "timeInForce": "GTC"}})

    def close_trade(self, trade_id, units="ALL"):
        data = {"units": units} if units != "ALL" else {}
        url = f"{self.base_url}/v3/accounts/{self.account_id}/trades/{trade_id}/close"
        resp = self.session.put(url, json=data, timeout=30)
        if resp.status_code == 200:
            logger.info(f"Trade {trade_id} closed")
            return resp.json()
        logger.error(f"Close trade {trade_id} failed: {resp.status_code} {resp.text}")
        return None

    def close_trade_partial(self, trade_id, units):
        """Close a portion of an open trade (partial close)."""
        return self.close_trade(trade_id, units=str(abs(units)))

    def get_open_trades(self):
        data = self._get(f"accounts/{self.account_id}/openTrades")
        if not data or "trades" not in data: return None
        return data["trades"]

    def get_trade(self, trade_id):
        data = self._get(f"accounts/{self.account_id}/trades/{trade_id}")
        if not data or "trade" not in data: return None
        return data["trade"]

    def get_open_positions(self):
        data = self._get(f"accounts/{self.account_id}/openPositions")
        if not data or "positions" not in data: return None
        return data["positions"]

    def test_connection(self) -> bool:
        try:
            result = self.get_account_summary()
            if result and "account" in result:
                acct = result["account"]
                logger.info(f"Connected to OANDA: Balance={acct.get('balance')}, NAV={acct.get('NAV')}, Currency={acct.get('currency')}")
                return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
        return False
