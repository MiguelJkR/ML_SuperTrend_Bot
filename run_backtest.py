"""
Run backtest for all active instruments with the ML SuperTrend v51 strategy.
Tests H1, M30, and M15 timeframes with fixed take-profit at 2:1 R:R.
Outputs results to console, saves JSON, and generates interactive HTML report.
"""
import sys
import json
import logging
import os
sys.path.insert(0, '.')

logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logging.getLogger('backtester').setLevel(logging.INFO)

from config import (OANDA_DEMO, STRATEGY, 
                    EUR_USD_M30, EUR_USD_M15,
                    GBP_USD_M30, GBP_USD_M15,
                    USD_JPY_M30, USD_JPY_M15)
from oanda_client import OandaClient
from backtester import Backtester
from backtest_report import generate_html_report

client = OandaClient(OANDA_DEMO)
bt = Backtester(client, STRATEGY)

# Backtest period: last 6 months
START = "2025-10-01T00:00:00Z"
END = "2026-04-08T00:00:00Z"
BALANCE = 200.0  # $200 USD (matches DEMO_CAPITAL_LIMIT)

# Test profitable timeframes + M15 candidate
INSTRUMENTS_TO_TEST = [
    EUR_USD_M30,       # PRIMARY
    EUR_USD_M15,       # Secondary
    GBP_USD_M30,       # NEW
    GBP_USD_M15,       # NEW
    USD_JPY_M30,       # NEW
    USD_JPY_M15,       # NEW
]

# CLI argument to filter
import argparse
parser = argparse.ArgumentParser(description="ML SuperTrend v51 Backtester")
parser.add_argument("--pair", type=str, help="Filter by pair e.g. EUR_USD")
parser.add_argument("--tf", type=str, help="Filter by timeframe e.g. M30")
parser.add_argument("--start", type=str, default=START, help="Start date")
parser.add_argument("--end", type=str, default=END, help="End date")
parser.add_argument("--balance", type=float, default=BALANCE, help="Starting balance")
args = parser.parse_args()

if args.start != START: START = args.start
if args.end != END: END = args.end
if args.balance != BALANCE: BALANCE = args.balance

if args.pair:
    INSTRUMENTS_TO_TEST = [i for i in INSTRUMENTS_TO_TEST if args.pair.upper() in i.symbol]
if args.tf:
    INSTRUMENTS_TO_TEST = [i for i in INSTRUMENTS_TO_TEST if args.tf.upper() == i.timeframe]

if not INSTRUMENTS_TO_TEST:
    print("No instruments match the filter. Available: EUR_USD, GBP_USD, USD_JPY (M30/M15)")
    sys.exit(1)

all_results = {}

for inst in INSTRUMENTS_TO_TEST:
    key = f"{inst.symbol}_{inst.timeframe}"
    print(f"\n{'='*80}")
    print(f"BACKTESTING: {key} | {START[:10]} to {END[:10]} | TP={inst.tp_rr_ratio}:1 R:R")
    print(f"{'='*80}")
    
    try:
        result = bt.run(inst, START, END, initial_balance=BALANCE)
        bt.print_summary(result)

        # Count exits by reason
        tp_exits = sum(1 for t in result.trades if t.exit_reason == "TP")
        sl_exits = sum(1 for t in result.trades if t.exit_reason == "SL")
        trail_exits = sum(1 for t in result.trades if t.exit_reason in ("TRAIL", "BE_SL"))
        end_exits = sum(1 for t in result.trades if t.exit_reason == "END")

        print(f"EXIT BREAKDOWN: TP={tp_exits}, SL={sl_exits}, Trail={trail_exits}, End={end_exits}")
        print(f"TP Rate: {tp_exits/max(1,result.total_trades):.1%}")
        
        # Serialize trade details for HTML report
        trades_detail = []
        for t in result.trades:
            trades_detail.append({
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": round(t.pnl, 4),
                "pnl_pct": round(t.pnl_pct, 2),
                "bars_held": t.bars_held,
                "exit_reason": t.exit_reason,
                "max_favorable_move": round(t.max_favorable_move, 6),
                "max_adverse_move": round(t.max_adverse_move, 6),
            })
        
        # Store for JSON export & HTML report
        all_results[key] = {
            "timeframe": inst.timeframe,
            "total_trades": result.total_trades,
            "wins": result.wins,
            "losses": result.losses,
            "win_rate": round(result.win_rate * 100, 2) if result.win_rate <= 1 else round(result.win_rate, 2),
            "profit_factor": round(result.profit_factor, 2),
            "total_pnl": round(result.total_pnl, 2),
            "avg_win": round(result.avg_win, 2),
            "avg_loss": round(result.avg_loss, 2),
            "largest_win": round(result.largest_win, 2),
            "largest_loss": round(result.largest_loss, 2),
            "max_drawdown_pct": round(result.max_drawdown_pct, 2),
            "sharpe_ratio": round(result.sharpe_ratio, 2),
            "sortino_ratio": round(getattr(result, "sortino_ratio", 0), 2),
            "calmar_ratio": round(getattr(result, "calmar_ratio", 0), 2),
            "expectancy": round(getattr(result, "expectancy", 0), 2),
            "avg_bars_held": round(getattr(result, "avg_bars_held", 0), 1),
            "return_pct": round(result.return_pct, 2),
            "initial_balance": result.initial_balance,
            "final_balance": round(result.final_balance, 2),
            "bars_tested": result.bars_tested,
            "max_consecutive_losses": result.consecutive_losses,
            "tp_rr_ratio": inst.tp_rr_ratio,
            "exit_breakdown": {"TP": tp_exits, "SL": sl_exits, "Trail": trail_exits, "End": end_exits},
            "equity_curve": [(t, round(b, 2)) for t, b in result.equity_curve[::max(1, len(result.equity_curve)//200)]],
            "trades_detail": trades_detail,
        }
    except Exception as e:
        print(f"ERROR backtesting {key}: {e}")
        import traceback
        traceback.print_exc()

# Save results to JSON
with open("backtest_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Print comparison summary
print(f"\n{'='*80}")
print("COMPARISON SUMMARY")
print(f"{'='*80}")
print(f"{'Instrument':<20} {'Trades':>7} {'Win%':>7} {'PF':>7} {'Return%':>9} {'MaxDD%':>8} {'TP Exits':>9}")
print("-" * 80)
for key, r in all_results.items():
    tp_pct = f"{r['exit_breakdown']['TP']}/{r['total_trades']}" if r['total_trades'] > 0 else "0/0"
    print(f"{key:<20} {r['total_trades']:>7} {r['win_rate']:>6.1f}% {r['profit_factor']:>7.2f} {r['return_pct']:>+8.2f}% {r['max_drawdown_pct']:>7.2f}% {tp_pct:>9}")
print("=" * 80)

# Generate HTML report
print("\nGenerating interactive HTML report...")
report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_report.html")
try:
    generate_html_report(all_results, report_path)
    print(f"HTML Report: {report_path}")
    print("Open in browser to see interactive charts!")
except Exception as e:
    print(f"Warning: Could not generate HTML report: {e}")

print(f"\nResults saved to backtest_results.json")
print("BACKTEST COMPLETE")
