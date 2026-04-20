"""
ML SuperTrend v51 - Broker Factory
====================================

Creates broker clients based on configuration.
Supports: OANDA (Forex/Gold), Binance (Crypto), Bybit (Crypto).
"""

import logging
from typing import Dict
from broker_base import BrokerClient

logger = logging.getLogger(__name__)


def create_broker(broker_type: str, config: dict) -> BrokerClient:
    """
    Create a broker client based on type.
    
    Args:
        broker_type: "OANDA", "BINANCE", or "BYBIT"
        config: Broker-specific configuration dict
    
    Returns:
        BrokerClient instance
    
    Raises:
        ValueError: If broker_type is not supported
        ImportError: If required library is not installed
    """
    broker_type = broker_type.upper()
    
    if broker_type == "OANDA":
        from broker_oanda import OandaBroker
        return OandaBroker(config)
    
    elif broker_type == "BINANCE":
        from broker_binance import BinanceBroker
        return BinanceBroker(config)
    
    elif broker_type == "BYBIT":
        from broker_bybit import BybitBroker
        return BybitBroker(config)
    
    else:
        raise ValueError(f"Unsupported broker type: {broker_type}. Use OANDA, BINANCE, or BYBIT.")


def create_all_brokers(broker_configs: Dict[str, dict]) -> Dict[str, BrokerClient]:
    """
    Create multiple broker clients from config dict.
    
    Args:
        broker_configs: {
            "OANDA": {"token": ..., "account_id": ..., ...},
            "BINANCE": {"api_key": ..., "api_secret": ..., "testnet": True},
            "BYBIT": {"api_key": ..., "api_secret": ..., "testnet": True},
        }
    
    Returns:
        Dict of {broker_name: BrokerClient}
    """
    brokers = {}
    for broker_type, config in broker_configs.items():
        try:
            broker = create_broker(broker_type, config)
            if broker.test_connection():
                brokers[broker_type] = broker
                logger.info(f"\u2713 {broker_type} broker connected successfully")
            else:
                logger.warning(f"\u2717 {broker_type} broker connection failed")
        except ImportError as e:
            logger.warning(f"\u2717 {broker_type} broker unavailable: {e}")
        except Exception as e:
            logger.error(f"\u2717 {broker_type} broker error: {e}")
    
    return brokers
