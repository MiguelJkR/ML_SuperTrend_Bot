"""
ML SuperTrend v51 - Broker Abstraction Layer
=============================================

Common interface for all brokers (OANDA, Binance, Bybit).
Each broker client must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from dataclasses import dataclass


@dataclass
class BrokerCandle:
    """Unified candle format across all brokers."""
    time: str        # ISO 8601
    open: float
    high: float
    low: float
    close: float
    volume: float
    complete: bool = True


@dataclass 
class BrokerPrice:
    """Unified price quote."""
    bid: float
    ask: float
    mid: float
    spread: float
    time: str


@dataclass
class BrokerOrderResult:
    """Unified order result."""
    success: bool
    order_id: str = ""
    fill_price: float = 0.0
    units: float = 0.0
    message: str = ""
    raw: dict = None
    
    def __post_init__(self):
        if self.raw is None:
            self.raw = {}


@dataclass
class BrokerTrade:
    """Unified open trade."""
    trade_id: str
    instrument: str
    direction: str      # "LONG" or "SHORT"
    units: float
    entry_price: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float
    open_time: str


class BrokerClient(ABC):
    """
    Abstract broker client. All broker implementations must follow this interface.
    This allows the strategy engine and trader to work with any broker.
    """
    
    @property
    @abstractmethod
    def broker_name(self) -> str:
        """Return broker name (e.g., 'OANDA', 'BINANCE', 'BYBIT')."""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test broker connectivity. Returns True if successful."""
        pass
    
    @abstractmethod
    def get_account_balance(self) -> float:
        """Return account balance in base currency."""
        pass
    
    @abstractmethod
    def get_candles(self, instrument: str, granularity: str, 
                    count: int = 300, from_time: str = None) -> Optional[List[Dict]]:
        """
        Fetch historical candles. Returns list of dicts with keys:
        time, open, high, low, close, volume, complete
        """
        pass
    
    @abstractmethod
    def get_current_price(self, instrument: str) -> Optional[Dict]:
        """
        Get current price. Returns dict with keys:
        bid, ask, mid, spread, time
        """
        pass
    
    @abstractmethod
    def market_order(self, instrument: str, units: float, 
                     sl_price: float = None, tp_price: float = None) -> Optional[BrokerOrderResult]:
        """
        Place a market order. Positive units = BUY, negative = SELL.
        """
        pass
    
    @abstractmethod
    def modify_trade_sl(self, trade_id: str, sl_price: float) -> bool:
        """Modify stop-loss on existing trade."""
        pass
    
    @abstractmethod
    def close_trade(self, trade_id: str, units: str = "ALL") -> bool:
        """Close a trade."""
        pass
    
    @abstractmethod
    def get_open_trades(self) -> Optional[List[BrokerTrade]]:
        """Get all open trades."""
        pass
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Convert unified symbol to broker-specific format.
        Override in subclass. Default: pass through.
        
        Unified: "BTC_USDT", "ETH_USDT", "EUR_USD"
        OANDA:   "EUR_USD" (same)
        Binance: "BTCUSDT"
        Bybit:   "BTCUSDT"
        """
        return symbol
    
    def denormalize_symbol(self, broker_symbol: str) -> str:
        """Convert broker-specific symbol back to unified format."""
        return broker_symbol
