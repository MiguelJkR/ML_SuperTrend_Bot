"""
OANDA v20 Streaming Price Client
=================================
Replaces REST polling with persistent streaming connection.
Provides real-time price ticks for faster trailing stop updates,
kill switch reactions, and spread monitoring.

OANDA uses chunked HTTP (not WebSocket) for streaming.
"""

import json
import time
import threading
import logging
import requests
from typing import Dict, List, Callable, Optional
from datetime import datetime, timezone
from collections import defaultdict

logger = logging.getLogger(__name__)


class OandaStreamClient:
    """
    Real-time price streaming from OANDA v20 API.
    
    Usage:
        stream = OandaStreamClient(config)
        stream.on_price(callback_function)
        stream.start(["EUR_USD", "GBP_USD", "USD_JPY"])
        # ... later ...
        stream.stop()
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Dict with keys: token, account_id, stream_url
        """
        self.token = config["token"]
        self.stream_url = config["stream_url"]
        self.account_id = config.get("account_id", "")
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        })
        
        # State
        self._running = False
        self._thread = None
        self._response = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_delay = 1.0  # seconds, doubles each retry up to 60s
        
        # Callbacks
        self._price_callbacks: List[Callable] = []
        self._heartbeat_callbacks: List[Callable] = []
        self._error_callbacks: List[Callable] = []
        
        # Latest prices cache
        self.latest_prices: Dict[str, Dict] = {}
        self.latest_spreads: Dict[str, float] = {}
        self.tick_counts: Dict[str, int] = defaultdict(int)
        self.last_heartbeat: Optional[datetime] = None
        
        # Stats
        self.total_ticks = 0
        self.start_time = None
        self.connection_drops = 0
        
        logger.info(f"OandaStreamClient initialized. Stream URL: {self.stream_url}")
    
    def on_price(self, callback: Callable):
        """Register a price update callback. Called with (instrument, bid, ask, time)."""
        self._price_callbacks.append(callback)
        return self
    
    def on_heartbeat(self, callback: Callable):
        """Register a heartbeat callback. Called with (timestamp)."""
        self._heartbeat_callbacks.append(callback)
        return self
    
    def on_error(self, callback: Callable):
        """Register an error callback. Called with (error_message)."""
        self._error_callbacks.append(callback)
        return self
    
    def start(self, instruments: List[str]):
        """
        Start streaming prices for given instruments.
        
        Args:
            instruments: List of OANDA instrument names, e.g. ["EUR_USD", "GBP_USD"]
        """
        if self._running:
            logger.warning("Stream already running. Stop first before restarting.")
            return
        
        self._running = True
        self._reconnect_attempts = 0
        self.start_time = datetime.now(timezone.utc)
        
        self._thread = threading.Thread(
            target=self._stream_loop,
            args=(instruments,),
            daemon=True,
            name="OandaPriceStream"
        )
        self._thread.start()
        logger.info(f"Price streaming started for: {', '.join(instruments)}")
    
    def stop(self):
        """Stop the streaming connection gracefully."""
        self._running = False
        
        if self._response:
            try:
                self._response.close()
            except:
                pass
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        
        logger.info(f"Price streaming stopped. Total ticks: {self.total_ticks}, "
                    f"Drops: {self.connection_drops}")
    
    def _stream_loop(self, instruments: List[str]):
        """Main streaming loop with automatic reconnection."""
        instruments_param = ",".join(instruments)
        
        while self._running:
            try:
                url = (f"{self.stream_url}/v3/accounts/{self.account_id}"
                       f"/pricing/stream?instruments={instruments_param}")
                
                logger.info(f"Connecting to price stream: {url}")
                
                self._response = self.session.get(
                    url,
                    stream=True,
                    timeout=(10, None)  # 10s connect timeout, no read timeout
                )
                
                if self._response.status_code != 200:
                    error_msg = f"Stream connection failed: {self._response.status_code}"
                    logger.error(error_msg)
                    self._notify_error(error_msg)
                    self._handle_reconnect()
                    continue
                
                # Reset reconnect counter on successful connection
                self._reconnect_attempts = 0
                self._reconnect_delay = 1.0
                logger.info("Price stream connected successfully")
                
                # Process streaming response line by line
                for line in self._response.iter_lines():
                    if not self._running:
                        break
                    
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line.decode('utf-8'))
                        self._process_message(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON in stream: {e}")
                    except Exception as e:
                        logger.error(f"Error processing stream message: {e}")
                
            except requests.exceptions.ConnectionError as e:
                if self._running:
                    logger.warning(f"Stream connection lost: {e}")
                    self.connection_drops += 1
                    self._handle_reconnect()
            except requests.exceptions.Timeout as e:
                if self._running:
                    logger.warning(f"Stream timeout: {e}")
                    self.connection_drops += 1
                    self._handle_reconnect()
            except Exception as e:
                if self._running:
                    logger.error(f"Unexpected stream error: {e}")
                    self.connection_drops += 1
                    self._handle_reconnect()
        
        logger.info("Stream loop exited")
    
    def _process_message(self, data: Dict):
        """Process a single streaming message."""
        msg_type = data.get("type", "")
        
        if msg_type == "PRICE":
            self._handle_price(data)
        elif msg_type == "HEARTBEAT":
            self._handle_heartbeat(data)
        else:
            logger.debug(f"Unknown stream message type: {msg_type}")
    
    def _handle_price(self, data: Dict):
        """Handle a price tick."""
        try:
            instrument = data.get("instrument", "UNKNOWN")
            
            # Extract bids and asks (OANDA sends arrays)
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            
            if not bids or not asks:
                return
            
            bid = float(bids[0].get("price", 0))
            ask = float(asks[0].get("price", 0))
            spread = ask - bid
            mid = (bid + ask) / 2.0
            tick_time = data.get("time", datetime.now(timezone.utc).isoformat())
            
            # Update cache
            self.latest_prices[instrument] = {
                'bid': bid,
                'ask': ask,
                'mid': mid,
                'spread': spread,
                'time': tick_time,
                'tradeable': data.get('tradeable', True),
            }
            self.latest_spreads[instrument] = spread
            self.tick_counts[instrument] += 1
            self.total_ticks += 1
            
            # Notify callbacks
            for cb in self._price_callbacks:
                try:
                    cb(instrument, bid, ask, tick_time)
                except Exception as e:
                    logger.error(f"Price callback error: {e}")
        
        except Exception as e:
            logger.error(f"Error handling price tick: {e}")
    
    def _handle_heartbeat(self, data: Dict):
        """Handle a heartbeat message."""
        self.last_heartbeat = datetime.now(timezone.utc)
        
        for cb in self._heartbeat_callbacks:
            try:
                cb(data.get("time", ""))
            except Exception as e:
                logger.error(f"Heartbeat callback error: {e}")
    
    def _handle_reconnect(self):
        """Handle reconnection with exponential backoff."""
        self._reconnect_attempts += 1
        
        if self._reconnect_attempts > self._max_reconnect_attempts:
            logger.error(f"Max reconnect attempts ({self._max_reconnect_attempts}) reached. Stopping stream.")
            self._running = False
            self._notify_error("Max reconnect attempts reached")
            return
        
        delay = min(self._reconnect_delay, 60.0)
        logger.info(f"Reconnecting in {delay:.1f}s (attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})")
        time.sleep(delay)
        self._reconnect_delay *= 2  # Exponential backoff
    
    def _notify_error(self, message: str):
        """Notify error callbacks."""
        for cb in self._error_callbacks:
            try:
                cb(message)
            except:
                pass
    
    def get_price(self, instrument: str) -> Optional[Dict]:
        """Get latest cached price for an instrument."""
        return self.latest_prices.get(instrument)
    
    def get_spread(self, instrument: str) -> float:
        """Get latest spread for an instrument."""
        return self.latest_spreads.get(instrument, 0.0)
    
    def get_stats(self) -> Dict:
        """Get streaming statistics."""
        uptime = 0
        if self.start_time:
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        tps = self.total_ticks / max(uptime, 1)
        
        return {
            'running': self._running,
            'total_ticks': self.total_ticks,
            'ticks_per_second': round(tps, 2),
            'tick_counts': dict(self.tick_counts),
            'connection_drops': self.connection_drops,
            'reconnect_attempts': self._reconnect_attempts,
            'uptime_seconds': round(uptime, 0),
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'instruments_streaming': list(self.latest_prices.keys()),
            'current_spreads': {k: round(v, 6) for k, v in self.latest_spreads.items()},
        }
    
    @property
    def is_connected(self) -> bool:
        """Check if stream is connected and receiving data."""
        if not self._running or not self.last_heartbeat:
            return False
        elapsed = (datetime.now(timezone.utc) - self.last_heartbeat).total_seconds()
        return elapsed < 10  # Heartbeats come every ~5s


# ============================================================
# STREAM MANAGER — integrates streaming with trading bot
# ============================================================
class StreamManager:
    """
    Manages streaming integration with the trading bot.
    Provides real-time spread monitoring, price alerts, and kill switch feed.
    """
    
    def __init__(self, config: dict, instruments: List[str]):
        self.stream = OandaStreamClient(config)
        self.instruments = instruments
        
        # Spread alert thresholds
        self.spread_alerts: Dict[str, float] = {}  # instrument -> max_spread
        self.spread_violations: List[Dict] = []
        
        # Price movement alerts (for kill switch)
        self.price_baselines: Dict[str, float] = {}  # instrument -> baseline price
        self.flash_crash_threshold: float = 0.005  # 0.5% sudden move
        self.flash_crash_alerts: List[Dict] = []
        
        # Register callbacks
        self.stream.on_price(self._on_price)
        self.stream.on_error(self._on_error)
        
        # External callbacks
        self._kill_switch_callback = None
        self._spread_alert_callback = None
    
    def set_spread_threshold(self, instrument: str, max_spread: float):
        """Set maximum acceptable spread for an instrument."""
        self.spread_alerts[instrument] = max_spread
    
    def on_kill_switch(self, callback: Callable):
        """Register callback for flash crash / kill switch events."""
        self._kill_switch_callback = callback
    
    def on_spread_alert(self, callback: Callable):
        """Register callback for spread violations."""
        self._spread_alert_callback = callback
    
    def start(self):
        """Start streaming."""
        self.stream.start(self.instruments)
    
    def stop(self):
        """Stop streaming."""
        self.stream.stop()
    
    def _on_price(self, instrument: str, bid: float, ask: float, time: str):
        """Handle price updates — check spreads and flash crashes."""
        spread = ask - bid
        mid = (bid + ask) / 2.0
        
        # Check spread threshold
        if instrument in self.spread_alerts:
            max_spread = self.spread_alerts[instrument]
            if spread > max_spread:
                alert = {
                    'instrument': instrument,
                    'spread': spread,
                    'threshold': max_spread,
                    'time': time,
                }
                self.spread_violations.append(alert)
                if len(self.spread_violations) > 100:
                    self.spread_violations = self.spread_violations[-100:]
                
                if self._spread_alert_callback:
                    try:
                        self._spread_alert_callback(alert)
                    except:
                        pass
        
        # Check flash crash (sudden large move)
        if instrument in self.price_baselines:
            baseline = self.price_baselines[instrument]
            if baseline > 0:
                pct_move = abs(mid - baseline) / baseline
                if pct_move > self.flash_crash_threshold:
                    alert = {
                        'instrument': instrument,
                        'baseline': baseline,
                        'current': mid,
                        'pct_move': pct_move * 100,
                        'time': time,
                    }
                    self.flash_crash_alerts.append(alert)
                    logger.warning(f"FLASH CRASH ALERT: {instrument} moved {pct_move*100:.2f}%!")
                    
                    if self._kill_switch_callback:
                        try:
                            self._kill_switch_callback(alert)
                        except:
                            pass
        
        # Update baseline (EMA-style, slow update to detect sudden moves)
        alpha = 0.001  # Very slow update
        if instrument in self.price_baselines:
            self.price_baselines[instrument] = (
                (1 - alpha) * self.price_baselines[instrument] + alpha * mid
            )
        else:
            self.price_baselines[instrument] = mid
    
    def _on_error(self, error_msg: str):
        """Handle stream errors."""
        logger.error(f"StreamManager error: {error_msg}")
    
    def get_stats(self) -> Dict:
        """Get combined stats."""
        return {
            'stream': self.stream.get_stats(),
            'spread_violations': len(self.spread_violations),
            'flash_crash_alerts': len(self.flash_crash_alerts),
            'recent_spread_violations': self.spread_violations[-5:],
            'recent_flash_crashes': self.flash_crash_alerts[-5:],
        }


# ============================================================
# EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    from config import OANDA_DEMO
    
    def on_price(instrument, bid, ask, time):
        spread = (ask - bid)
        print(f"[{instrument}] Bid: {bid:.5f} Ask: {ask:.5f} Spread: {spread:.5f}")
    
    def on_heartbeat(time):
        print(f"[HEARTBEAT] {time}")
    
    stream = OandaStreamClient(OANDA_DEMO)
    stream.on_price(on_price)
    stream.on_heartbeat(on_heartbeat)
    stream.start(["EUR_USD", "GBP_USD", "USD_JPY"])
    
    try:
        while True:
            time.sleep(10)
            print(f"\nStats: {json.dumps(stream.get_stats(), indent=2)}\n")
    except KeyboardInterrupt:
        stream.stop()
        print("Stream stopped.")
