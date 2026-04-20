""" 
ML SuperTrend v51 Trading Bot - Entry Point
Usage:
    python main.py              # Start bot (poll + dashboard)
    python main.py --test       # Test API connection only
    python main.py --poll-once  # Run one poll cycle and exit
    python main.py --dashboard  # Dashboard only (no trading)
"""

import sys
import os
import logging
import argparse
import threading
from config import (LOG_FILE, LOG_LEVEL, OANDA_LIVE, LIVE_TRADING_ENABLED,
                    ACTIVE_INSTRUMENTS, POLL_INTERVAL)

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


def test_connection():
    """Test OANDA API connection."""
    from oanda_client import OandaClient
    
    logger.info("=" * 60)
    logger.info("ML SuperTrend v51 Bot - Connection Test")
    logger.info("=" * 60)
    
    client = OandaClient(OANDA_LIVE)
    
    if client.test_connection():
        logger.info("API Connection: OK")
        
        # Test candle fetch for each instrument
        for inst in ACTIVE_INSTRUMENTS:
            candles = client.get_candles(inst.symbol, inst.timeframe, count=10)
            if candles:
                logger.info(f"  {inst.symbol} {inst.timeframe}: {len(candles)} candles OK "
                          f"(last close: {candles[-1]['close']})")
            else:
                logger.error(f"  {inst.symbol} {inst.timeframe}: FAILED to fetch candles")
        
        # Test price fetch
        for inst in ACTIVE_INSTRUMENTS:
            price = client.get_current_price(inst.symbol)
            if price:
                logger.info(f"  {inst.symbol} price: bid={price['bid']:.5f} "
                          f"ask={price['ask']:.5f} spread={price['spread']:.5f}")
            else:
                logger.error(f"  {inst.symbol}: FAILED to fetch price")
        
        logger.info("=" * 60)
        logger.info("All tests passed!")
        return True
    else:
        logger.error("API Connection: FAILED")
        return False


def run_bot(with_dashboard=True, paper_mode=False, demo_mode=False, dual_mode=False):
    """Start the trading bot."""
    from trader import Trader
    from telegram_bot import TelegramBot

    if dual_mode:
        mode = "DUAL (LIVE + DEMO)"
    elif demo_mode:
        mode = "DEMO ONLY"
    elif paper_mode:
        mode = "PAPER TRADING"
    else:
        mode = "LIVE" if LIVE_TRADING_ENABLED else "SIGNALS ONLY"

    logger.info("=" * 60)
    logger.info(f"ML SuperTrend v51 Trading Bot \u2014 {mode}")
    logger.info(f"Instruments: {[f'{i.symbol}_{i.timeframe}' for i in ACTIVE_INSTRUMENTS]}")
    logger.info(f"Poll Interval: {POLL_INTERVAL}s")
    logger.info("=" * 60)

    trader = Trader(paper_mode=paper_mode, demo_mode=demo_mode, dual_mode=dual_mode)
    
    # Setup telegram
    telegram = TelegramBot()
    if telegram.test_connection():
        trader.set_telegram(telegram)
        telegram.set_trader(trader)
        telegram.start_polling()
        logger.info('Telegram command polling active')
    
    # Start dashboard in background thread
    if with_dashboard:
        from dashboard import set_modules, run_dashboard
        set_modules(
            trader=trader,
            regime=getattr(trader, 'regime_v2', None),
            ml=getattr(trader, 'ml_scorer', None),
            advisor=getattr(trader, 'advisor', None),
            risk=getattr(trader, 'smart_risk', None),
            wfo=getattr(trader, 'wfo', None),
            rl_scorer=getattr(trader, 'rl_scorer', None),
            stream_manager=getattr(trader, 'stream_manager', None),
        )
        dash_thread = threading.Thread(target=run_dashboard, daemon=True)
        dash_thread.start()
        logger.info("Dashboard started in background")
    
    # Run main loop
    trader.run(POLL_INTERVAL)


def run_dashboard_only():
    """Start dashboard without trading."""
    from trader import Trader
    from dashboard import set_modules, run_dashboard
    
    trader = Trader()
    set_modules(
        trader=trader,
        regime=getattr(trader, 'regime_v2', None),
        ml=getattr(trader, 'ml_scorer', None),
        advisor=getattr(trader, 'advisor', None),
        risk=getattr(trader, 'smart_risk', None),
        wfo=getattr(trader, 'wfo', None),
        rl_scorer=getattr(trader, 'rl_scorer', None),
        stream_manager=getattr(trader, 'stream_manager', None),
    )

    logger.info("Dashboard-only mode (no trading)")
    run_dashboard()


def main():
    parser = argparse.ArgumentParser(description="ML SuperTrend v51 Trading Bot")
    parser.add_argument("--test", action="store_true", help="Test API connection")
    parser.add_argument("--paper", action="store_true", help="Paper trading mode (simulated)")
    parser.add_argument("--demo", action="store_true", help="Demo account only (fxpractice)")
    parser.add_argument("--dual", action="store_true", help="Dual mode: LIVE + DEMO in parallel")
    parser.add_argument("--poll-once", action="store_true", help="Run one poll cycle")
    parser.add_argument("--dashboard", action="store_true", help="Dashboard only")
    parser.add_argument("--no-dashboard", action="store_true", help="No dashboard")

    args = parser.parse_args()

    if args.test:
        success = test_connection()
        # Also test demo if available
        from config import OANDA_DEMO
        if OANDA_DEMO.get("token") and OANDA_DEMO["token"] != "YOUR_TOKEN":
            logger.info("\nTesting DEMO connection...")
            from oanda_client import OandaClient
            demo = OandaClient(OANDA_DEMO)
            if demo.test_connection():
                logger.info(f"DEMO Account ID: {demo.account_id}")
            else:
                logger.warning("DEMO connection failed")
        sys.exit(0 if success else 1)

    elif args.poll_once:
        from trader import Trader
        trader = Trader(demo_mode=args.demo)
        result = trader.poll_once()
        for key, data in result.items():
            logger.info(f"{key}: {data}")
        sys.exit(0)

    elif args.dashboard:
        run_dashboard_only()

    else:
        run_bot(
            with_dashboard=not args.no_dashboard,
            paper_mode=args.paper,
            demo_mode=args.demo,
            dual_mode=args.dual,
        )


if __name__ == "__main__":
    main()

