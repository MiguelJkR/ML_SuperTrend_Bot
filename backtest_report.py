"""
ML SuperTrend v51 - Advanced HTML Backtest Report Generator
============================================================

Generates interactive HTML reports with:
- Equity curve (Plotly)
- Drawdown chart
- Monthly returns heatmap
- Trade distribution (win/loss, by direction, by exit reason)
- Trade scatter (PnL vs bars held)
- Rolling win rate & profit factor
- Multi-strategy comparison table
- Individual trade log with filters

Uses Plotly.js CDN for interactive charts. Single self-contained HTML file.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional


def generate_html_report(
    results: Dict,
    output_path: str = "backtest_report.html",
    title: str = "ML SuperTrend v51 - Backtest Report"
) -> str:
    """
    Generate a comprehensive interactive HTML backtest report.
    
    Args:
        results: Dict of {instrument_key: result_dict} from run_backtest.py
        output_path: Path for the HTML file
        title: Report title
    
    Returns:
        Path to generated HTML file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build sections for each instrument
    instrument_sections = ""
    comparison_rows = ""
    
    for key, r in results.items():
        trades = r.get("trades_detail", [])
        equity = r.get("equity_curve", [])
        
        # --- Equity Curve ---
        eq_times = [e[0][:10] for e in equity] if equity else []
        eq_vals = [e[1] for e in equity] if equity else []
        
        # --- Drawdown ---
        dd_vals = []
        if eq_vals:
            peak = eq_vals[0]
            for v in eq_vals:
                peak = max(peak, v)
                dd_vals.append(((v - peak) / peak) * 100 if peak > 0 else 0)
        
        # --- Monthly Returns ---
        monthly = _compute_monthly_returns(trades)
        
        # --- Trade PnL list ---
        trade_pnls = [t["pnl"] for t in trades] if trades else []
        trade_dirs = [t["direction"] for t in trades] if trades else []
        trade_exits = [t["exit_reason"] for t in trades] if trades else []
        trade_bars = [t["bars_held"] for t in trades] if trades else []
        trade_times = [t["entry_time"][:10] for t in trades] if trades else []
        
        # Rolling stats (window=10)
        rolling_wr = _rolling_win_rate(trade_pnls, window=10)
        rolling_pf = _rolling_profit_factor(trade_pnls, window=10)
        
        # Exit breakdown
        eb = r.get("exit_breakdown", {})
        
        instrument_sections += _build_instrument_section(
            key=key,
            r=r,
            eq_times=json.dumps(eq_times),
            eq_vals=json.dumps(eq_vals),
            dd_vals=json.dumps(dd_vals),
            monthly=json.dumps(monthly),
            trade_pnls=json.dumps(trade_pnls),
            trade_dirs=json.dumps(trade_dirs),
            trade_exits=json.dumps(trade_exits),
            trade_bars=json.dumps(trade_bars),
            trade_times=json.dumps(trade_times),
            rolling_wr=json.dumps(rolling_wr),
            rolling_pf=json.dumps(rolling_pf),
            trades=trades,
            eb=eb,
        )
        
        # Comparison row
        tp_pct = f"{eb.get('TP',0)}/{r['total_trades']}" if r['total_trades'] > 0 else "0/0"
        comparison_rows += f"""
        <tr>
            <td><strong>{key}</strong></td>
            <td>{r['total_trades']}</td>
            <td>{r['win_rate']:.1f}%</td>
            <td>{r['profit_factor']:.2f}</td>
            <td class="{'positive' if r['return_pct']>0 else 'negative'}">{r['return_pct']:+.2f}%</td>
            <td>${r['total_pnl']:+.2f}</td>
            <td>{r['max_drawdown_pct']:.2f}%</td>
            <td>{r['sharpe_ratio']:.2f}</td>
            <td>{r['avg_win']:+.2f} / {r['avg_loss']:+.2f}</td>
            <td>{tp_pct}</td>
            <td>{r.get('max_consecutive_losses', 0)}</td>
        </tr>"""
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1923; color: #e0e0e0; padding: 20px;
        }}
        h1 {{ color: #00d4aa; text-align: center; padding: 20px 0; font-size: 1.8em; }}
        h2 {{ color: #00b894; margin: 30px 0 15px 0; padding-bottom: 8px; 
              border-bottom: 2px solid #1e3a4f; font-size: 1.4em; }}
        h3 {{ color: #74b9ff; margin: 20px 0 10px 0; font-size: 1.1em; }}
        .header-info {{ text-align: center; color: #636e72; margin-bottom: 30px; }}
        .section {{ background: #162736; border-radius: 12px; padding: 25px; 
                   margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; }}
        .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
        .stat-card {{
            background: #1e3a4f; border-radius: 8px; padding: 15px; text-align: center;
        }}
        .stat-card .label {{ color: #636e72; font-size: 0.85em; margin-bottom: 5px; }}
        .stat-card .value {{ font-size: 1.5em; font-weight: bold; }}
        .positive {{ color: #00d4aa; }}
        .negative {{ color: #ff6b6b; }}
        .neutral {{ color: #74b9ff; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 0.9em; }}
        th {{ background: #1e3a4f; color: #74b9ff; padding: 10px 8px; text-align: left; 
             position: sticky; top: 0; }}
        td {{ padding: 8px; border-bottom: 1px solid #1e3a4f; }}
        tr:hover {{ background: rgba(116, 185, 255, 0.05); }}
        .chart {{ width: 100%; min-height: 350px; }}
        .chart-small {{ width: 100%; min-height: 280px; }}
        .trade-log {{ max-height: 400px; overflow-y: auto; }}
        .badge {{ display: inline-block; padding: 3px 10px; border-radius: 4px; 
                 font-size: 0.8em; font-weight: bold; }}
        .badge-long {{ background: rgba(0,212,170,0.2); color: #00d4aa; }}
        .badge-short {{ background: rgba(255,107,107,0.2); color: #ff6b6b; }}
        .badge-tp {{ background: rgba(0,212,170,0.2); color: #00d4aa; }}
        .badge-sl {{ background: rgba(255,107,107,0.2); color: #ff6b6b; }}
        .badge-trail {{ background: rgba(116,185,255,0.2); color: #74b9ff; }}
        .badge-end {{ background: rgba(99,110,114,0.2); color: #636e72; }}
        .tab-buttons {{ display: flex; gap: 5px; margin-bottom: 15px; flex-wrap: wrap; }}
        .tab-btn {{ background: #1e3a4f; color: #74b9ff; border: none; padding: 8px 16px;
                   border-radius: 6px; cursor: pointer; font-size: 0.9em; }}
        .tab-btn:hover {{ background: #2d5a7f; }}
        .tab-btn.active {{ background: #00b894; color: #0f1923; }}
        .instrument-section {{ display: none; }}
        .instrument-section.active {{ display: block; }}
        @media (max-width: 768px) {{
            .grid-2, .grid-3, .grid-4 {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <h1>📊 {title}</h1>
    <div class="header-info">Generated: {timestamp} | Period: varies per instrument</div>
    
    <!-- COMPARISON TABLE -->
    <div class="section">
        <h2>📋 Multi-Strategy Comparison</h2>
        <div style="overflow-x: auto;">
        <table>
            <thead>
                <tr>
                    <th>Instrument</th><th>Trades</th><th>Win%</th><th>PF</th>
                    <th>Return</th><th>PnL</th><th>Max DD</th><th>Sharpe</th>
                    <th>Avg W/L</th><th>TP Exits</th><th>Max ConsL</th>
                </tr>
            </thead>
            <tbody>{comparison_rows}</tbody>
        </table>
        </div>
    </div>
    
    <!-- TAB BUTTONS -->
    <div class="tab-buttons" id="tabButtons"></div>
    
    <!-- INSTRUMENT SECTIONS -->
    {instrument_sections}
    
    <script>
    // Tab navigation
    const keys = {json.dumps(list(results.keys()))};
    const tabContainer = document.getElementById('tabButtons');
    keys.forEach((k, i) => {{
        const btn = document.createElement('button');
        btn.className = 'tab-btn' + (i === 0 ? ' active' : '');
        btn.textContent = k;
        btn.onclick = () => {{
            document.querySelectorAll('.instrument-section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('section-' + k).classList.add('active');
            btn.classList.add('active');
            // Resize plotly charts
            setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
        }};
        tabContainer.appendChild(btn);
    }});
    if (keys.length > 0) {{
        document.getElementById('section-' + keys[0]).classList.add('active');
    }}
    </script>
</body>
</html>"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    return output_path


def _build_instrument_section(key, r, eq_times, eq_vals, dd_vals, monthly,
                               trade_pnls, trade_dirs, trade_exits, trade_bars,
                               trade_times, rolling_wr, rolling_pf, trades, eb):
    """Build HTML section for one instrument."""
    
    ret_class = "positive" if r['return_pct'] > 0 else "negative"
    pf_class = "positive" if r['profit_factor'] > 1 else "negative"
    
    # Trade log rows
    trade_rows = ""
    for t in trades:
        dir_badge = "badge-long" if t["direction"] == "LONG" else "badge-short"
        exit_badge = f"badge-{t['exit_reason'].lower()}"
        pnl_class = "positive" if t["pnl"] > 0 else "negative"
        trade_rows += f"""
        <tr>
            <td>{t['entry_time'][:16]}</td>
            <td>{t['exit_time'][:16]}</td>
            <td><span class="badge {dir_badge}">{t['direction']}</span></td>
            <td>{t['entry_price']:.5f}</td>
            <td>{t['exit_price']:.5f}</td>
            <td class="{pnl_class}">${t['pnl']:+.2f}</td>
            <td class="{pnl_class}">{t['pnl_pct']:+.2f}%</td>
            <td>{t['bars_held']}</td>
            <td><span class="badge {exit_badge}">{t['exit_reason']}</span></td>
        </tr>"""
    
    chart_id = key.replace("/", "_").replace(" ", "_")
    
    return f"""
    <div class="instrument-section" id="section-{key}">
        <!-- KPI Cards -->
        <div class="section">
            <h2>📈 {key} — Performance Summary</h2>
            <div class="grid-4">
                <div class="stat-card">
                    <div class="label">Total Return</div>
                    <div class="value {ret_class}">{r['return_pct']:+.2f}%</div>
                </div>
                <div class="stat-card">
                    <div class="label">Total PnL</div>
                    <div class="value {ret_class}">${r['total_pnl']:+.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Win Rate</div>
                    <div class="value neutral">{r['win_rate']:.1f}%</div>
                </div>
                <div class="stat-card">
                    <div class="label">Profit Factor</div>
                    <div class="value {pf_class}">{r['profit_factor']:.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Total Trades</div>
                    <div class="value">{r['total_trades']}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Sharpe Ratio</div>
                    <div class="value neutral">{r['sharpe_ratio']:.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Max Drawdown</div>
                    <div class="value negative">{r['max_drawdown_pct']:.2f}%</div>
                </div>
                <div class="stat-card">
                    <div class="label">Max Consec. Losses</div>
                    <div class="value negative">{r.get('max_consecutive_losses', 0)}</div>
                </div>
            </div>
        </div>
        
        <!-- Equity Curve -->
        <div class="section">
            <h3>Equity Curve</h3>
            <div id="equity_{chart_id}" class="chart"></div>
        </div>
        
        <!-- Drawdown -->
        <div class="section">
            <h3>Drawdown %</h3>
            <div id="dd_{chart_id}" class="chart-small"></div>
        </div>
        
        <!-- Charts Row -->
        <div class="grid-2">
            <div class="section">
                <h3>PnL Distribution</h3>
                <div id="pnl_hist_{chart_id}" class="chart-small"></div>
            </div>
            <div class="section">
                <h3>Exit Reason Breakdown</h3>
                <div id="exit_pie_{chart_id}" class="chart-small"></div>
            </div>
        </div>
        
        <div class="grid-2">
            <div class="section">
                <h3>PnL vs Bars Held</h3>
                <div id="scatter_{chart_id}" class="chart-small"></div>
            </div>
            <div class="section">
                <h3>Cumulative PnL by Trade #</h3>
                <div id="cum_pnl_{chart_id}" class="chart-small"></div>
            </div>
        </div>
        
        <div class="grid-2">
            <div class="section">
                <h3>Rolling Win Rate (10-trade)</h3>
                <div id="roll_wr_{chart_id}" class="chart-small"></div>
            </div>
            <div class="section">
                <h3>Rolling Profit Factor (10-trade)</h3>
                <div id="roll_pf_{chart_id}" class="chart-small"></div>
            </div>
        </div>
        
        <!-- Monthly Heatmap -->
        <div class="section">
            <h3>Monthly Returns Heatmap</h3>
            <div id="monthly_{chart_id}" class="chart"></div>
        </div>
        
        <!-- Trade Log -->
        <div class="section">
            <h3>Trade Log ({r['total_trades']} trades)</h3>
            <div class="trade-log">
                <table>
                    <thead>
                        <tr>
                            <th>Entry</th><th>Exit</th><th>Dir</th><th>Entry$</th>
                            <th>Exit$</th><th>PnL</th><th>PnL%</th><th>Bars</th><th>Reason</th>
                        </tr>
                    </thead>
                    <tbody>{trade_rows}</tbody>
                </table>
            </div>
        </div>
        
        <script>
        (function() {{
            const layout_dark = {{
                paper_bgcolor: '#162736', plot_bgcolor: '#0f1923',
                font: {{ color: '#e0e0e0', size: 11 }},
                margin: {{ t: 30, r: 20, b: 40, l: 60 }},
                xaxis: {{ gridcolor: '#1e3a4f' }},
                yaxis: {{ gridcolor: '#1e3a4f' }},
            }};
            const config = {{ responsive: true, displayModeBar: false }};
            
            // Equity Curve
            Plotly.newPlot('equity_{chart_id}', [{{
                x: {eq_times}, y: {eq_vals}, type: 'scatter', mode: 'lines',
                fill: 'tozeroy', fillcolor: 'rgba(0,212,170,0.1)',
                line: {{ color: '#00d4aa', width: 2 }}, name: 'Equity'
            }}], {{...layout_dark, yaxis: {{...layout_dark.yaxis, title: 'Balance ($)'}}}}, config);
            
            // Drawdown
            Plotly.newPlot('dd_{chart_id}', [{{
                x: {eq_times}, y: {dd_vals}, type: 'scatter', mode: 'lines',
                fill: 'tozeroy', fillcolor: 'rgba(255,107,107,0.15)',
                line: {{ color: '#ff6b6b', width: 1.5 }}, name: 'Drawdown'
            }}], {{...layout_dark, yaxis: {{...layout_dark.yaxis, title: 'Drawdown %'}}}}, config);
            
            // PnL Histogram
            const pnls = {trade_pnls};
            Plotly.newPlot('pnl_hist_{chart_id}', [{{
                x: pnls, type: 'histogram', nbinsx: 25,
                marker: {{ color: pnls.map(p => p > 0 ? '#00d4aa' : '#ff6b6b') }},
            }}], {{...layout_dark, xaxis: {{...layout_dark.xaxis, title: 'PnL ($)'}}}}, config);
            
            // Exit Pie
            const exits = {trade_exits};
            const exitCounts = {{}};
            exits.forEach(e => exitCounts[e] = (exitCounts[e]||0)+1);
            const exitColors = {{'TP': '#00d4aa', 'SL': '#ff6b6b', 'TRAIL': '#74b9ff', 'BE_SL': '#fdcb6e', 'END': '#636e72'}};
            Plotly.newPlot('exit_pie_{chart_id}', [{{
                labels: Object.keys(exitCounts), values: Object.values(exitCounts),
                type: 'pie', hole: 0.45,
                marker: {{ colors: Object.keys(exitCounts).map(k => exitColors[k] || '#636e72') }},
                textinfo: 'label+percent',
            }}], {{...layout_dark}}, config);
            
            // PnL vs Bars Scatter
            const bars = {trade_bars};
            const dirs = {trade_dirs};
            Plotly.newPlot('scatter_{chart_id}', [{{
                x: bars, y: pnls, mode: 'markers', type: 'scatter',
                marker: {{
                    color: pnls.map(p => p > 0 ? '#00d4aa' : '#ff6b6b'),
                    size: 7, opacity: 0.7,
                    symbol: dirs.map(d => d === 'LONG' ? 'triangle-up' : 'triangle-down'),
                }},
                text: dirs.map((d,i) => d + ' $' + pnls[i].toFixed(2)),
            }}], {{...layout_dark, xaxis: {{...layout_dark.xaxis, title: 'Bars Held'}},
                   yaxis: {{...layout_dark.yaxis, title: 'PnL ($)'}}}}, config);
            
            // Cumulative PnL
            let cum = 0;
            const cumPnl = pnls.map(p => {{ cum += p; return cum; }});
            Plotly.newPlot('cum_pnl_{chart_id}', [{{
                y: cumPnl, type: 'scatter', mode: 'lines',
                line: {{ color: '#74b9ff', width: 2 }},
            }}], {{...layout_dark, xaxis: {{...layout_dark.xaxis, title: 'Trade #'}},
                   yaxis: {{...layout_dark.yaxis, title: 'Cumulative PnL ($)'}}}}, config);
            
            // Rolling WR
            Plotly.newPlot('roll_wr_{chart_id}', [{{
                y: {rolling_wr}, type: 'scatter', mode: 'lines',
                line: {{ color: '#fdcb6e', width: 2 }},
            }}, {{
                y: Array({r['total_trades']}).fill(50), type: 'scatter', mode: 'lines',
                line: {{ color: '#636e72', width: 1, dash: 'dash' }}, name: '50%'
            }}], {{...layout_dark, yaxis: {{...layout_dark.yaxis, title: 'Win Rate %'}}}}, config);
            
            // Rolling PF
            Plotly.newPlot('roll_pf_{chart_id}', [{{
                y: {rolling_pf}, type: 'scatter', mode: 'lines',
                line: {{ color: '#a29bfe', width: 2 }},
            }}, {{
                y: Array({r['total_trades']}).fill(1.0), type: 'scatter', mode: 'lines',
                line: {{ color: '#636e72', width: 1, dash: 'dash' }}, name: 'PF=1.0'
            }}], {{...layout_dark, yaxis: {{...layout_dark.yaxis, title: 'Profit Factor'}}}}, config);
            
            // Monthly Heatmap
            const monthly = {monthly};
            if (monthly.months && monthly.months.length > 0) {{
                Plotly.newPlot('monthly_{chart_id}', [{{
                    x: monthly.months, y: ['Return %'], z: [monthly.returns],
                    type: 'heatmap', colorscale: [
                        [0, '#ff6b6b'], [0.5, '#162736'], [1, '#00d4aa']
                    ],
                    text: [monthly.returns.map(v => v.toFixed(2) + '%')],
                    texttemplate: '%{{text}}', textfont: {{ size: 11 }},
                    showscale: true, zmid: 0,
                }}], {{...layout_dark, yaxis: {{autorange: true}}}}, config);
            }}
        }})();
        </script>
    </div>"""


def _compute_monthly_returns(trades):
    """Compute monthly returns from trade list."""
    if not trades:
        return {"months": [], "returns": []}
    
    monthly = {}
    for t in trades:
        try:
            month = t["entry_time"][:7]  # "YYYY-MM"
            monthly[month] = monthly.get(month, 0) + t["pnl"]
        except (KeyError, TypeError):
            continue
    
    sorted_months = sorted(monthly.keys())
    return {
        "months": sorted_months,
        "returns": [round(monthly[m], 2) for m in sorted_months]
    }


def _rolling_win_rate(pnls, window=10):
    """Compute rolling win rate over last N trades."""
    if len(pnls) < window:
        return [sum(1 for p in pnls[:i+1] if p > 0) / (i+1) * 100 for i in range(len(pnls))]
    
    result = []
    for i in range(len(pnls)):
        start = max(0, i - window + 1)
        chunk = pnls[start:i+1]
        wr = sum(1 for p in chunk if p > 0) / len(chunk) * 100
        result.append(round(wr, 1))
    return result


def _rolling_profit_factor(pnls, window=10):
    """Compute rolling profit factor over last N trades."""
    result = []
    for i in range(len(pnls)):
        start = max(0, i - window + 1)
        chunk = pnls[start:i+1]
        wins = sum(p for p in chunk if p > 0)
        losses = abs(sum(p for p in chunk if p < 0))
        pf = wins / losses if losses > 0 else (10.0 if wins > 0 else 0)
        result.append(round(min(pf, 10.0), 2))
    return result


if __name__ == "__main__":
    # Quick test: load from existing JSON
    import sys
    json_path = "backtest_results.json"
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        
        report_path = generate_html_report(data, "backtest_report.html")
        print(f"Report generated: {report_path}")
    else:
        print(f"No results file found at {json_path}")
        print("Run run_backtest.py first to generate results.")
