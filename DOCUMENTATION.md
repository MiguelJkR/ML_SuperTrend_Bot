# ML SuperTrend v51 — Complete Technical Documentation

> **Algorithmic Trading Bot with 20 Scientific ML Strategies**
> GPU-Optimized (RTX 4070 Super + i7-14700KF) | Multi-Broker | Real-Time Streaming

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [File Structure & Module Map](#3-file-structure--module-map)
4. [Core Trading Engine](#4-core-trading-engine)
5. [Broker Abstraction Layer](#5-broker-abstraction-layer)
6. [Technical Indicators](#6-technical-indicators)
7. [Risk Management System](#7-risk-management-system)
8. [ML Models — Classical](#8-ml-models--classical)
9. [ML Models — Deep Learning (LSTM)](#9-ml-models--deep-learning-lstm)
10. [ML Models — Reinforcement Learning](#10-ml-models--reinforcement-learning)
11. [20 Scientific Learning Strategies](#11-20-scientific-learning-strategies)
12. [Regime Detection](#12-regime-detection)
13. [Feature Engineering & Causal Analysis](#13-feature-engineering--causal-analysis)
14. [Sentiment & News Analysis](#14-sentiment--news-analysis)
15. [Market Data Enrichment](#15-market-data-enrichment)
16. [Multi-Timeframe Analysis](#16-multi-timeframe-analysis)
17. [Portfolio Optimization](#17-portfolio-optimization)
18. [Backtesting Engine](#18-backtesting-engine)
19. [Optimization & Training Pipeline](#19-optimization--training-pipeline)
20. [Experiment Tracking](#20-experiment-tracking)
21. [Smart Alerts System](#21-smart-alerts-system)
22. [Dashboard & Monitoring](#22-dashboard--monitoring)
23. [Telegram Bot](#23-telegram-bot)
24. [Docker Deployment](#24-docker-deployment)
25. [Configuration Reference](#25-configuration-reference)
26. [Mathematical Formulas Reference](#26-mathematical-formulas-reference)
27. [Academic References](#27-academic-references)
28. [Installation & Setup](#28-installation--setup)

---

## 1. Project Overview

ML SuperTrend v51 is a production-grade algorithmic trading bot that combines classical technical analysis (SuperTrend, RSI, ADX, MACD) with 20 cutting-edge machine learning strategies spanning deep learning, reinforcement learning, Bayesian inference, causal analysis, and portfolio theory.

**Key Characteristics:**

- **29,100+ lines** of Python across **59 modules**
- **20 scientific ML strategies** with academic paper backing
- **Multi-broker**: OANDA v20, Binance Futures, Bybit V5
- **GPU-accelerated**: PyTorch CUDA 12.x on RTX 4070 Super
- **Graceful degradation**: Every module wraps imports in `try/except` — the bot runs with zero ML dependencies if needed
- **Real-time streaming**: OANDA v20 chunked HTTP price streaming
- **Continuous learning**: Nightly retraining (11 PM), weekly review (Friday 5 PM)
- **Risk-first**: Kelly Criterion, HMM regime gating, drawdown protection, correlation limits

**Trading Instruments:**

| Instrument | Timeframes | Broker |
|---|---|---|
| EUR/USD | M15, M30, H1 | OANDA |
| GBP/USD | M15, M30 | OANDA |
| USD/JPY | M15, M30 | OANDA |
| XAU/USD | M30, H1 | OANDA |
| BTC/USDT | M30, H1 | Binance/Bybit |
| ETH/USDT | H1 | Binance/Bybit |
| SOL/USDT | H1 | Binance/Bybit |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ML SuperTrend v51                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐              │
│  │  main.py │───▶│  trader.py   │───▶│ strategy.py   │              │
│  │ (entry)  │    │(orchestrator)│    │ (signal gen)  │              │
│  └──────────┘    └──────┬───────┘    └───────────────┘              │
│                         │                                           │
│         ┌───────────────┼───────────────────┐                       │
│         ▼               ▼                   ▼                       │
│  ┌─────────────┐ ┌─────────────┐  ┌──────────────────┐             │
│  │  BROKERS    │ │  ML MODELS  │  │  RISK MANAGEMENT │             │
│  │             │ │             │  │                  │             │
│  │ oanda_client│ │ lstm_predict│  │ risk_manager     │             │
│  │ oanda_stream│ │ ensemble    │  │ smart_risk       │             │
│  │ crypto_client│ │ rl_scorer  │  │ kelly_sizing     │             │
│  │ broker_*    │ │ ml_learner  │  │ portfolio_opt    │             │
│  └─────────────┘ │ deep_models │  │ correlation_mgr  │             │
│                  │ mtf_lstm    │  └──────────────────┘             │
│                  └─────────────┘                                   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │              SCIENTIFIC STRATEGIES (20)                   │       │
│  │                                                          │       │
│  │  advanced_learning.py:                                   │       │
│  │    SharpeLoss, QuantileHead, MCDropout, TDLambda,       │       │
│  │    FisherDetector, CurriculumScheduler, FinancialPE,    │       │
│  │    EXP3OnlineLearner                                    │       │
│  │                                                          │       │
│  │  deep_models.py:                                        │       │
│  │    VSN, ContrastiveLearner, MarketVAE, KnowledgeDistill,│       │
│  │    MAMLTrainer, CrossAssetGNN                           │       │
│  │                                                          │       │
│  │  causal_features.py:                                    │       │
│  │    GrangerCausality, WassersteinDrift, InfoBottleneck   │       │
│  │                                                          │       │
│  │  wavelet_features.py, hmm_regime.py, data_augmentation  │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                     │
│  ┌────────────┐ ┌──────────────┐ ┌─────────────┐ ┌──────────┐     │
│  │ dashboard  │ │ telegram_bot │ │ backtester  │ │ training │     │
│  │ (Flask API)│ │ (2-way chat) │ │ (Monte Carlo│ │ _manager │     │
│  │ + HTML v2  │ │              │ │  + Stress)  │ │ (nightly)│     │
│  └────────────┘ └──────────────┘ └─────────────┘ └──────────┘     │
│                                                                     │
│  ┌──────────────┐ ┌────────────────┐ ┌──────────────────────┐      │
│  │ smart_alerts │ │ experiment_    │ │ walk_forward_v2     │      │
│  │ (10 types +  │ │ tracker        │ │ (overfitting detect) │      │
│  │  cooldown)   │ │ (leaderboard)  │ │                      │      │
│  └──────────────┘ └────────────────┘ └──────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

**Data Flow:**

```
Market Data → Indicators → Strategy Signals → ML Ensemble Scoring
    │                                              │
    ▼                                              ▼
Regime Detection (HMM)                    Confidence Filter
    │                                              │
    ▼                                              ▼
Session/News Filter                       Risk Management
    │                                     (Kelly + Drawdown)
    ▼                                              │
Correlation Check ◄────────────────────────────────┘
    │
    ▼
Order Execution (OANDA/Binance/Bybit)
    │
    ▼
Trade Management (Trailing SL, Breakeven, TP)
    │
    ▼
Performance → Experiment Tracker → Training Manager → Model Update
```

---

## 3. File Structure & Module Map

```
ML_SuperTrend_Bot/
│
├── main.py                    # Entry point (CLI: --demo, --paper, --dual)
├── trader.py                  # Main orchestrator (900+ lines)
├── config.py                  # All configuration (instruments, API keys, params)
│
├── ── BROKERS ──
├── oanda_client.py            # OANDA v20 REST API client
├── oanda_stream.py            # OANDA v20 streaming (chunked HTTP)
├── crypto_client.py           # Binance/Bybit unified client (raw HTTP)
├── broker_base.py             # Abstract broker interface
├── broker_oanda.py            # OANDA adapter
├── broker_binance.py          # Binance Futures adapter
├── broker_bybit.py            # Bybit V5 adapter
├── broker_factory.py          # Factory pattern for broker creation
│
├── ── STRATEGY ──
├── strategy.py                # Signal generation (SuperTrend + filters)
├── indicators.py              # ATR, EMA, SMA, RSI, ADX, MACD, SuperTrend
├── market_structure.py        # S/R levels, regime detection (BBW+ADX)
├── mtf_analyzer.py            # Multi-timeframe confirmation (H1/H4/D1)
│
├── ── RISK MANAGEMENT ──
├── risk_manager.py            # Trade management (breakeven, trailing, SL)
├── smart_risk.py              # Dynamic position sizing by conditions
├── kelly_sizing.py            # Kelly Criterion + 6 dynamic multipliers
├── portfolio_optimizer.py     # Markowitz, Risk Parity, Max Sharpe
├── correlation_manager.py     # Cross-pair exposure limits
├── session_filter.py          # Trading session awareness
│
├── ── ML MODELS ──
├── ml_learner.py              # Logistic Regression + XGBoost
├── ensemble_scorer.py         # Weighted ensemble (soft voting + EXP3)
├── rl_scorer.py               # DQN Reinforcement Learning (numpy)
├── lstm_predictor.py          # LSTM + Multi-Head Attention (PyTorch/GPU)
├── mtf_lstm.py                # Multi-Timeframe LSTM (M5/M15/H1/H4)
├── feature_engine.py          # Auto feature engineering + anomaly detection
│
├── ── SCIENTIFIC STRATEGIES ──
├── advanced_learning.py       # Strategies 1-8 (Sharpe Loss, MC Dropout, etc.)
├── deep_models.py             # Strategies 9-14 (VSN, VAE, GNN, MAML, etc.)
├── causal_features.py         # Strategies 15-17 (Granger, Wasserstein, InfoBottleneck)
├── wavelet_features.py        # Strategy 16: Wavelet Decomposition (DWT)
├── hmm_regime.py              # Strategy 7: HMM Regime Detection
├── data_augmentation.py       # Strategy 6: Time Series Augmentation
│
├── ── SENTIMENT & DATA ──
├── sentiment.py               # OANDA order/position book sentiment
├── sentiment_engine.py        # FinBERT NLP sentiment (HuggingFace)
├── news_feed.py               # RSS + NewsAPI + Finnhub live feed
├── news_filter.py             # Economic calendar (ForexFactory)
├── market_data_enricher.py    # yfinance + FRED macroeconomic data
├── macro_filter.py            # Fed balance sheet, yield curve
│
├── ── REGIME DETECTION ──
├── regime_detector.py         # ADX + BBW + Hurst exponent
├── hmm_regime.py              # Hidden Markov Model (3-state)
│
├── ── BACKTESTING & VALIDATION ──
├── backtester.py              # Professional backtester (Monte Carlo, Stress)
├── walk_forward.py            # Walk-Forward Optimizer v1 (grid search)
├── walk_forward_v2.py         # Walk-Forward Validator v2 (overfitting)
├── run_backtest.py            # CLI backtest runner
├── run_backtest_comparison.py # Before/after comparison
├── backtest_report.py         # Interactive HTML report (Plotly)
│
├── ── OPTIMIZATION & TRAINING ──
├── hyperopt.py                # Optuna Bayesian optimization
├── auto_optimizer.py          # Weekly grid search optimization
├── training_manager.py        # Continuous learning (nightly + weekly)
├── experiment_tracker.py      # Experiment logging + leaderboard
│
├── ── INTERFACE & MONITORING ──
├── dashboard.py               # Flask API (35+ endpoints)
├── dashboard_v2.html          # Interactive Chart.js dashboard
├── telegram_bot.py            # Two-way Telegram bot
├── smart_alerts.py            # 10 alert types + cooldown anti-spam
├── financial_advisor.py       # Spanish-language AI advisor (FinBot)
│
├── ── PAPER TRADING ──
├── paper_trader.py            # Simulated trading engine
│
├── ── COLAB TRAINING ──
├── colab_trainer.py           # Google Colab notebook generator
├── colab_notebook_cells.py    # Notebook cell definitions
│
├── ── TESTING ──
├── test_v5_complete.py        # Comprehensive test suite
├── diagnose_signals.py        # Signal filter diagnostics
│
├── ── DEPLOYMENT ──
├── Dockerfile                 # PyTorch CUDA 12.4 container
├── docker-compose.yml         # Bot + dashboard + GPU passthrough
├── .dockerignore
├── requirements.txt           # CPU-only dependencies
├── requirements_gpu.txt       # GPU dependencies (CUDA 12.x)
│
└── ── GENERATED DATA ──
    ├── experiments/           # Experiment JSON logs
    ├── models/                # Saved model weights
    ├── data/                  # Cache + historical data
    └── logs/                  # Application logs
```

**Module count by category:**

| Category | Files | Lines (approx) |
|---|---|---|
| Brokers | 7 | ~2,500 |
| Strategy & Indicators | 3 | ~1,800 |
| Risk Management | 5 | ~2,200 |
| ML Models | 6 | ~4,500 |
| Scientific Strategies | 6 | ~4,000 |
| Sentiment & Data | 6 | ~2,800 |
| Backtesting | 5 | ~2,400 |
| Optimization & Training | 4 | ~2,200 |
| Interface & Monitoring | 5 | ~3,500 |
| Core (trader, main, config) | 3 | ~2,000 |
| Other (paper, colab, test) | 9 | ~1,200 |
| **Total** | **59** | **~29,100** |

---

## 4. Core Trading Engine

### 4.1 Main Entry Point (`main.py`)

```bash
python main.py --demo          # Demo mode (OANDA practice)
python main.py --paper         # Paper trading (simulated)
python main.py --dual          # Live + Demo in parallel
python main.py --test          # Connection test only
python main.py --dashboard     # Dashboard server only
python main.py --poll-once     # Single polling cycle
```

### 4.2 Orchestrator (`trader.py`)

The `Trader` class is the central hub that:

1. Initializes all modules with graceful degradation (40+ `try/except` import blocks)
2. Runs the polling loop (`poll_once()` every 30 seconds)
3. Coordinates the signal pipeline: Candles → Indicators → Strategy → Filters → ML Scoring → Risk Check → Execution
4. Manages trade lifecycle: entry, breakeven, trailing stop, take profit, exit

**Module initialization order (in `__init__`):**

```
v1: OandaClient, StrategyEngine, RiskManager
v2: MTFAnalyzer, NewsFilter, SentimentAnalyzer, RegimeDetector, SRLevels
v3: MarketRegimeDetector, MLTradeScorer, FinancialAdvisor, SmartRiskManager, WalkForwardOptimizer
v4: DQNTradeScorer, OandaStreamClient
v5: CryptoClient, CorrelationManager, SessionFilter, FeatureEngineer, LSTMPredictor, EnsembleScorer
v6: MarketDataEnricher, SentimentEngine, NewsFeed
v7: TrainingManager, HMMRegimeDetector, FisherChangeDetector
v8: WaveletDecomposer, GrangerCausalitySelector, WassersteinDriftDetector, MarketVAE, CrossAssetGNN
v9: Backtester, SmartAlertManager, HyperOptimizer, MultiTimeframeLSTM
v10: KellySizer, WalkForwardValidator, ExperimentTracker, PortfolioOptimizer
```

### 4.3 Strategy Engine (`strategy.py`)

Generates trade signals based on SuperTrend crossovers with multiple confirmation filters.

**Signal Generation Pipeline:**

```
SuperTrend Direction Change
    │
    ├── RSI Filter (avoid overbought/oversold)
    ├── ADX Filter (minimum trend strength)
    ├── ATR Volatility Spike Filter
    ├── SuperTrend Distance Filter (not too far from entry)
    ├── Confirmation Candle (wait 1 bar)
    └── Minimum Signal Strength Threshold
    │
    ▼
Signal { direction, entry_price, stop_loss, take_profit, strength }
```

**Signal Dataclass:**

```python
@dataclass
class Signal:
    direction: str          # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float        # ATR-based
    take_profit: float      # Adaptive R:R ratio
    strength: float         # 0.0 - 1.0
    mtf_score: float        # Multi-timeframe confirmation
    regime: str             # Market regime context
    sentiment_bias: float   # Sentiment overlay
```

---

## 5. Broker Abstraction Layer

### 5.1 Architecture

```
BrokerClient (Abstract Base)
    │
    ├── OandaBroker     → OandaClient (OANDA v20 REST)
    ├── BinanceBroker   → CryptoClient (Binance USDT-M Futures)
    └── BybitBroker     → CryptoClient (Bybit V5 API)
```

### 5.2 Unified Interface (`broker_base.py`)

```python
class BrokerClient(ABC):
    def get_candles(symbol, granularity, count) -> List[BrokerCandle]
    def get_current_price(symbol) -> BrokerPrice
    def place_market_order(symbol, units, sl, tp) -> BrokerOrderResult
    def close_trade(trade_id) -> BrokerOrderResult
    def get_open_trades() -> List[BrokerTrade]
    def get_account_summary() -> Dict
    def modify_trade(trade_id, sl, tp) -> BrokerOrderResult
```

### 5.3 OANDA Streaming (`oanda_stream.py`)

Real-time price streaming using OANDA v20 chunked HTTP:

- Persistent connection with auto-reconnect (exponential backoff)
- Real-time spread monitoring with per-instrument thresholds
- Trailing stop updates on every tick
- Kill switch for emergency position closure

---

## 6. Technical Indicators

### `indicators.py` — Core Functions

| Function | Description | Formula |
|---|---|---|
| `atr(h, l, c, p)` | Average True Range | `RMA(max(H-L, |H-Cp|, |L-Cp|), p)` |
| `ema(data, p)` | Exponential Moving Average | `α = 2/(p+1)` |
| `sma(data, p)` | Simple Moving Average | `mean(data[-p:])` |
| `rma(data, p)` | Running Moving Average | Wilder's smoothing |
| `rsi(closes, p)` | Relative Strength Index | `100 - 100/(1 + avg_gain/avg_loss)` |
| `adx(h, l, c, p)` | Average Directional Index | Based on +DI/-DI |
| `macd(c, f, s, sig)` | MACD | `EMA(f) - EMA(s)`, signal = `EMA(MACD, sig)` |
| `supertrend(h,l,c,p,f)` | SuperTrend | `ATR×factor ± HL2` with direction flip |
| `kmeans_volatility_clustering()` | Volatility clustering | K-means on ATR percentiles |
| `compute_all_indicators()` | All-in-one | Returns dict with all indicators |

---

## 7. Risk Management System

### 7.1 Multi-Layer Architecture

```
Layer 1: Pre-Trade Filters
    ├── Session Filter (exchange hours, day of week)
    ├── News Filter (economic calendar)
    ├── Correlation Manager (group exposure limits)
    ├── Max Daily Trades (10/day)
    └── Max Daily Loss (12% of equity)

Layer 2: Position Sizing
    ├── Kelly Criterion (dynamic, Half-Kelly)
    ├── Smart Risk (regime-adjusted)
    └── Portfolio Optimization (Markowitz weights)

Layer 3: Trade Management
    ├── Initial Stop Loss (ATR-based)
    ├── Breakeven Trigger
    ├── Trailing Stop (adaptive)
    └── Take Profit (R:R ratio)

Layer 4: Portfolio Risk
    ├── Max Concurrent Positions (3)
    ├── Correlation Group Limits (2% per group)
    └── Drawdown Protection (progressive size reduction)
```

### 7.2 Kelly Criterion (`kelly_sizing.py`)

**Classic Kelly:**

```
f* = (p × b - q) / b

where:
  p = P(win)           — win rate
  b = avg_win/avg_loss — payoff ratio
  q = 1 - p            — loss rate
```

**Half-Kelly (default, more conservative):**

```
f_half = f* / 2
```

**6 Dynamic Adjustment Multipliers:**

```
f_adjusted = f_half × regime × uncertainty × drift × wavelet × confidence × drawdown
```

| Multiplier | Source | Values |
|---|---|---|
| `regime` | HMM Regime Detector | TRENDING=1.0, MEAN_REVERT=0.7, VOLATILE=0.4 |
| `uncertainty` | MC Dropout | should_trade=True→1.0, False→0.3 |
| `drift` | Wasserstein Distance | NORMAL=1.0, DRIFT=0.7, CRITICAL=0.4 |
| `wavelet` | Wavelet Noise Level | <35%=1.0, 35-50%=0.7, >50%=0.5 |
| `confidence` | Signal Strength | Interpolated [0.5, 1.0] |
| `drawdown` | Current Drawdown | <1.5%=1.0, 1.5-3%=0.75, 3-5%=0.5, >5%=0.3 |

**Position Size Bounds:** min 0.25%, max 3.0% of capital.

### 7.3 Correlation Manager (`correlation_manager.py`)

Pre-defined correlation groups:

```python
CORRELATION_GROUPS = {
    "USD_NEGATIVE": ["EUR_USD", "GBP_USD", "AUD_USD"],  # Inversely correlated with USD
    "USD_POSITIVE": ["USD_JPY", "USD_CHF", "USD_CAD"],   # Positively correlated with USD
    "COMMODITIES": ["XAU_USD", "XAG_USD"],                # Precious metals
    "CRYPTO_MAJOR": ["BTC_USDT", "ETH_USDT", "SOL_USDT"] # Crypto basket
}
```

**Rules:**
- Max 2% total risk per correlated group
- Max 2 same-direction trades in any group
- Inverse correlation pairs: block opposing signals

---

## 8. ML Models — Classical

### 8.1 Logistic Regression (`ml_learner.py`)

Pure numpy implementation (no sklearn dependency):

```python
class LogisticRegressionLearner:
    # Sigmoid: σ(z) = 1 / (1 + e^(-z))
    # Loss: -[y·log(ŷ) + (1-y)·log(1-ŷ)]
    # Update: w -= lr × ∇L
```

**Features (TradeFeatures):**

```
atr_ratio, rsi_norm, adx_norm, macd_hist_norm, bb_position,
volume_ratio, trend_strength, hour_sin, hour_cos, day_sin, day_cos,
consecutive_wins, consecutive_losses, recent_win_rate
```

### 8.2 XGBoost Scorer (`ml_learner.py`)

Gradient boosted trees for trade quality prediction:

```python
class XGBoostTradeScorer:
    # n_estimators=100, max_depth=4, learning_rate=0.1
    # Binary classification: TAKE vs SKIP
```

### 8.3 Ensemble Scorer (`ensemble_scorer.py`)

Weighted soft voting with adaptive weights:

```python
class EnsembleScorer:
    initial_weights = {
        "logistic_regression": 0.30,
        "xgboost": 0.40,
        "dqn_rl": 0.30
    }
    agreement_threshold = 0.60     # Minimum consensus
    high_confidence_threshold = 0.72
    low_confidence_threshold = 0.40
```

**Integration with EXP3:** Weights are dynamically adjusted by the EXP3 online learner based on each model's recent performance.

---

## 9. ML Models — Deep Learning (LSTM)

### 9.1 Architecture (`lstm_predictor.py`)

```
Input (seq_length=30, features=14+embeddings)
    │
    ▼
┌──────────────────────────────┐
│  LSTM (128 hidden, 2 layers) │  ← GPU: PyTorch CUDA
│  + Dropout (0.3)             │  ← CPU: NumpyGRU fallback
└──────────────┬───────────────┘
               │
    ▼
┌──────────────────────────────┐
│  Multi-Head Temporal         │
│  Attention (4 heads, d_k=32) │
│  Q = W_q × H                │
│  K = W_k × H                │
│  V = W_v × H                │
│  Attn = softmax(QK'/√d_k) V │
└──────────────┬───────────────┘
               │
    ▼
┌──────────────────────────────┐
│  Classifier Head             │
│  Linear(128→64) → ReLU      │
│  → Dropout → Linear(64→3)   │
│  → Softmax                   │
│  Classes: [DOWN, FLAT, UP]   │
└──────────────────────────────┘
```

**GPU Configuration (RTX 4070 Super):**

```python
GPU_CONFIG = {
    "hidden_size": 128,
    "num_layers": 2,
    "n_heads": 4,
    "batch_size": 64,
    "use_amp": True,      # Automatic Mixed Precision (FP16)
    "sequence_length": 30,
    "features": 14
}
```

**Embeddings:**
- Instrument ID → 4-dim embedding (8 instruments)
- Trading Session → 3-dim embedding (Asian/London/NY/Off)

### 9.2 Multi-Timeframe LSTM (`mtf_lstm.py`)

Processes 4 timeframes in parallel with cross-temporal attention fusion:

```
M5  → TimeframeEncoder (LSTM 64h) ──┐
M15 → TimeframeEncoder (LSTM 64h) ──┤
H1  → TimeframeEncoder (LSTM 64h) ──┼──▶ CrossTemporalAttention (4-head MHA)
H4  → TimeframeEncoder (LSTM 64h) ──┘         │
                                               ▼
                                    Learnable TF Importance
                                    (softmax weights per TF)
                                               │
                                               ▼
                                    Prediction Head → [DOWN, FLAT, UP]
```

---

## 10. ML Models — Reinforcement Learning

### 10.1 DQN Trade Scorer (`rl_scorer.py`)

Deep Q-Network (pure numpy) that learns whether to TAKE or SKIP signals:

```
State: [14 trade features]
Actions: {TAKE, SKIP}
Reward: PnL of trade outcome (normalized)

Q(s,a) = r + γ × max_a' Q(s', a')

Architecture:
  Input(14) → Dense(64) → ReLU → Dense(32) → ReLU → Dense(2)

Training:
  - Prioritized Experience Replay (α=0.6, β=0.4→1.0)
  - Target network (soft update τ=0.01)
  - ε-greedy exploration (1.0 → 0.05, decay=0.995)
```

---

## 11. 20 Scientific Learning Strategies

### Strategy 1: Sharpe-Aware Loss Function

**Paper:** Moody & Saffell (2001)

```
L_sharpe = -S(R) = -(E[R] / σ(R))

Differentiable approximation:
  A_t = A_{t-1} + η(R_t - A_t)        # EMA of returns
  B_t = B_{t-1} + η(R_t² - B_t)       # EMA of squared returns
  S_t = A_t / √(B_t - A_t²)
```

**File:** `advanced_learning.py` → `SharpeLoss`

### Strategy 2: TD-λ (Temporal Difference Learning)

**Paper:** Sutton (1988)

```
V(s_t) ← V(s_t) + α × δ_t × e_t

where:
  δ_t = r_t + γV(s_{t+1}) - V(s_t)    # TD error
  e_t = γλe_{t-1} + ∇V(s_t)            # Eligibility trace

λ = 0.8 (trace decay)
γ = 0.95 (discount factor)
```

**File:** `advanced_learning.py` → `TDLambdaEvaluator`

### Strategy 3: HMM Regime Detection

**Paper:** Hamilton (1989), Guidolin & Timmermann (2007)

```
3-state Gaussian HMM:
  States: {TRENDING, MEAN_REVERTING, VOLATILE}
  
  Observations: [returns, volatility, volume]
  
  Transition matrix A:
    P(S_{t+1}=j | S_t=i) = a_ij
  
  Emission: B(o_t | S_t=j) = N(μ_j, σ_j²)
  
  Decoding: Viterbi algorithm
  Training: Baum-Welch (EM)
```

**File:** `hmm_regime.py` → `HMMRegimeDetector`
**Fallback:** `NumpyHMM` (pure numpy Baum-Welch when hmmlearn unavailable)

### Strategy 4: Quantile Regression

**Paper:** Koenker & Bassett (1978)

```
Pinball Loss:
  ρ_τ(u) = u(τ - I(u<0))

Quantile predictions at τ = {0.1, 0.25, 0.5, 0.75, 0.9}
  → Prediction intervals for risk assessment
  → IQR = Q75 - Q25 (uncertainty measure)
```

**File:** `advanced_learning.py` → `QuantileHead`

### Strategy 5: EXP3 Online Learning

**Paper:** Auer et al. (2002)

```
Exponential-weight algorithm for Exploration and Exploitation:

  w_i(t+1) = w_i(t) × exp(γ × r̂_i(t) / K)

where:
  r̂_i = r_i / p_i(t)                   # Importance-weighted reward
  p_i(t) = (1-γ) × w_i/Σw + γ/K       # Mixed strategy
  γ = min(1, √(K ln K / ((e-1)T)))     # Exploration rate
```

**File:** `advanced_learning.py` → `EXP3OnlineLearner`
**Usage:** Dynamically adjusts ensemble model weights

### Strategy 6: Data Augmentation

**Paper:** Um et al. (2017)

6 augmentation techniques for time series:

| Technique | Description |
|---|---|
| Jitter | Add Gaussian noise N(0, σ²) |
| Magnitude Warp | Smooth random scaling via cubic spline |
| Time Warp | Temporal distortion via cubic spline |
| Window Slice | Random sub-window crop |
| Permutation | Shuffle temporal segments |
| Mixup | Linear interpolation between samples |

**File:** `data_augmentation.py` → `TimeSeriesAugmentor`

### Strategy 7: Fisher Information Change Detection

**Paper:** Ly et al. (2017)

```
Fisher Information Matrix:
  I(θ) = E[∇log p(x|θ) × ∇log p(x|θ)ᵀ]

Approximation (for neural nets):
  I ≈ (1/N) Σ ∇L(x_i, θ)²

Change score:
  s_t = ||I_t - I_{t-k}|| / ||I_{t-k}||

Alerts:
  s_t > 2.0 → WARNING (regime shift starting)
  s_t > 3.0 → CRITICAL (significant regime change)
```

**File:** `advanced_learning.py` → `FisherChangeDetector`

### Strategy 8: Curriculum Learning

**Paper:** Bengio et al. (2009)

```
Phase 1 (Easy):     Low volatility, trending periods
Phase 2 (Medium):   Normal market conditions
Phase 3 (Hard):     High volatility, regime transitions, news events
Phase 4 (Full):     All data equally

Difficulty scoring:
  d = α × volatility_rank + β × regime_uncertainty + γ × news_proximity
```

**File:** `advanced_learning.py` → `CurriculumScheduler`

### Strategy 9: MC Dropout (Bayesian Uncertainty)

**Paper:** Gal & Ghahramani (2016)

```
Monte Carlo Dropout for uncertainty estimation:

  1. Keep dropout ON at inference time
  2. Run T=20 forward passes
  3. Predictions: {ŷ_1, ŷ_2, ..., ŷ_T}
  4. Mean prediction: μ = (1/T) Σ ŷ_t
  5. Uncertainty: σ = std(ŷ_1, ..., ŷ_T)

Decision:
  σ < threshold → should_trade = True (confident)
  σ ≥ threshold → should_trade = False (uncertain, reduce size to 30%)
```

**File:** `advanced_learning.py` → `MCDropoutPredictor`

### Strategy 10: Financial Positional Encoding

**Paper:** Inspired by Vaswani et al. (2017), adapted for markets

```
4 temporal components:
  1. Sinusoidal position: sin(pos / 10000^(2i/d))
  2. Time-of-day: sin(2π × hour/24)
  3. Day-of-week: sin(2π × day/5)
  4. Month-of-year: sin(2π × month/12)

Combined: PE = W_1×pos + W_2×tod + W_3×dow + W_4×moy
```

**File:** `advanced_learning.py` → `FinancialPositionalEncoding`

### Strategy 11: Temporal Fusion Transformer (VSN)

**Paper:** Lim et al. (2021)

```
Variable Selection Network:
  1. Per-feature GRN: h_i = GRN(x_i)
  2. Softmax gating: v = softmax(GRN(concat(x_1,...,x_n)))
  3. Selected features: x̃ = Σ v_i × h_i

Gated Residual Network (GRN):
  η_1 = W_1 × x + b_1
  η_2 = W_2 × ELU(η_1) + b_2
  GLU = σ(η_2[:d]) ⊙ η_2[d:]
  output = LayerNorm(x + GLU)
```

**File:** `deep_models.py` → `VariableSelectionNetwork`, `GatedResidualNetwork`

### Strategy 12: Wasserstein Distance (Drift Detection)

**Paper:** Vallender (1974), Ramdas et al. (2017)

```
Earth Mover's Distance between distributions:
  W_1(P, Q) = inf_{γ∈Γ(P,Q)} E_{(x,y)~γ}[|x - y|]

Simplified (1D sorted):
  W_1 = (1/n) Σ |F_P^{-1}(i/n) - F_Q^{-1}(i/n)|

Monitoring:
  reference_window = 200 bars (baseline distribution)
  test_window = 50 bars (current distribution)
  
  W < 0.5 → NORMAL
  0.5 ≤ W < 1.0 → DRIFT (reduce position to 70%)
  W ≥ 1.0 → CRITICAL (reduce to 40%, trigger retraining)
```

**File:** `causal_features.py` → `WassersteinDriftDetector`

### Strategy 13: Contrastive Learning (TS2Vec)

**Paper:** Yue et al. (2022)

```
InfoNCE Loss:
  L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))

where:
  z_i, z_j = augmented views of same sample (positive pair)
  z_k = different samples (negative pairs)
  sim(a,b) = a·b / (||a||×||b||) (cosine similarity)
  τ = 0.07 (temperature)
```

**File:** `deep_models.py` → `ContrastiveLearner`

### Strategy 14: Knowledge Distillation

**Paper:** Hinton et al. (2015)

```
Teacher-Student Framework:
  L_distill = α × KL(softmax(z_t/T), softmax(z_s/T)) + (1-α) × L_CE(y, z_s)

where:
  z_t = teacher logits (large model)
  z_s = student logits (small model)
  T = temperature (2-20)
  α = 0.7 (distillation weight)
```

**File:** `deep_models.py` → `KnowledgeDistiller`

### Strategy 15: MAML (Model-Agnostic Meta-Learning)

**Paper:** Finn et al. (2017)

```
Meta-Learning for rapid regime adaptation:

Inner loop (task-specific):
  θ'_i = θ - α × ∇_θ L_Ti(f_θ)

Outer loop (meta-update):
  θ ← θ - β × Σ_i ∇_θ L_Ti(f_{θ'_i})

Tasks = different market regimes
Goal: learn θ that adapts to new regime in few gradient steps
```

**File:** `deep_models.py` → `MAMLTrainer`

### Strategy 16: Wavelet Decomposition

**Paper:** Daubechies (1992), Ramsey & Lampart (1998)

```
Discrete Wavelet Transform (DWT):
  Wavelet: Daubechies-4 (db4)
  Levels: 4

  Level 1: High-frequency noise (1-2 bars)
  Level 2: Short-term momentum (2-4 bars)
  Level 3: Medium-term cycles (4-8 bars)
  Level 4: Trend component (8-16 bars)
  Approximation: Long-term trend (16+ bars)

Denoising:
  1. Decompose signal
  2. Apply soft thresholding to detail coefficients
  3. Reconstruct → clean signal

Energy distribution:
  E_level = Σ |c_level|² / Σ |c_all|²
```

**File:** `wavelet_features.py` → `WaveletDecomposer`

### Strategy 17: VAE (Variational Autoencoder)

**Paper:** Kingma & Welling (2014)

```
Market anomaly detection via latent space:

Encoder: x → μ, log(σ²)
  z = μ + σ × ε,  ε ~ N(0,1)  (reparameterization trick)

Decoder: z → x̂

ELBO Loss:
  L = ||x - x̂||² + β × KL(q(z|x) || p(z))
  KL = -0.5 × Σ(1 + log(σ²) - μ² - σ²)

Anomaly score:
  a = ||x - x̂||²  (reconstruction error)
  a > threshold → anomalous market regime
```

**File:** `deep_models.py` → `MarketVAE`

### Strategy 18: Graph Neural Network (Cross-Asset)

**Paper:** Velickovic et al. (2018) — Graph Attention Networks

```
Cross-asset dependency modeling:

  Attention coefficients:
    e_ij = LeakyReLU(a^T [W×h_i || W×h_j])
    α_ij = softmax_j(e_ij)

  Node update:
    h'_i = σ(Σ_j α_ij × W × h_j)

  Multi-head (4 heads):
    h'_i = ||_{k=1}^K σ(Σ_j α_ij^k × W^k × h_j)

Nodes = assets (8: EUR/USD, GBP/USD, USD/JPY, XAU, BTC, ETH, SOL, SPX)
Edges = learned correlations
```

**File:** `deep_models.py` → `CrossAssetGNN`

### Strategy 19: Information Bottleneck

**Paper:** Tishby et al. (2000)

```
Feature compression minimizing:
  L = I(X; Z) - β × I(Z; Y)

where:
  X = input features
  Z = compressed representation
  Y = target (trade outcome)
  β = trade-off parameter

Implementation:
  VAE-style bottleneck layer:
    X (n_features) → encoder → Z (bottleneck_dim) → decoder → X̂
  
  Leave-one-out importance:
    importance_i = |loss_full - loss_without_i|
```

**File:** `causal_features.py` → `InformationBottleneckSelector`

### Strategy 20: Granger Causality

**Paper:** Granger (1969)

```
Feature X Granger-causes Y if:
  Var(Y_t | Y_{t-1},...,Y_{t-p}) > Var(Y_t | Y_{t-1},...,Y_{t-p}, X_{t-1},...,X_{t-p})

F-test:
  F = ((RSS_restricted - RSS_unrestricted) / p) / (RSS_unrestricted / (n - 2p - 1))

p-value < 0.05 → X is a causal predictor of Y
```

**File:** `causal_features.py` → `GrangerCausalitySelector`

---

## 12. Regime Detection

### 12.1 Three-Layer Regime System

```
Layer 1: RegimeDetector (market_structure.py)
  - Bollinger Band Width + ADX → RANGING / TRENDING / BREAKOUT

Layer 2: MarketRegimeDetector (regime_detector.py)
  - ADX + BBW + ATR + EMA + Hurst exponent
  - States: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE

Layer 3: HMMRegimeDetector (hmm_regime.py)
  - 3-state Gaussian HMM (hmmlearn or numpy fallback)
  - States: TRENDING, MEAN_REVERTING, VOLATILE
  - Viterbi decoding + forward-backward for state probabilities
```

### 12.2 Regime Impact on Trading

| Regime | Position Size | Strategy | SL Width |
|---|---|---|---|
| TRENDING | 100% (Kelly) | Follow trend | Normal |
| MEAN_REVERTING | 70% | Counter-trend, tighter TP | Wider |
| VOLATILE | 40% | Reduce exposure | 1.5x wider |

---

## 13. Feature Engineering & Causal Analysis

### 13.1 Auto Feature Engineering (`feature_engine.py`)

```python
class FeatureEngineer:
    # Generates 50+ features per candle:
    # - Interaction features (RSI × ADX, MACD × volume)
    # - Z-scores (price, volume, ATR)
    # - Rate of change (5, 10, 20 bar)
    # - Percentile ranks (rolling 50-bar)
    # - Candle patterns (doji, hammer, engulfing)
    # - Time features (hour_sin, hour_cos, day_sin, day_cos)
```

### 13.2 Anomaly Detection (`feature_engine.py`)

```python
class AnomalyDetector:
    # Isolation Forest (sklearn)
    # contamination=0.05 (5% expected anomalies)
    # Refit every 200 bars
    # Score: -1 (anomaly) to +1 (normal)
```

### 13.3 Causal Feature Selection (`causal_features.py`)

Three methods for identifying truly predictive features:

1. **Granger Causality** — F-test for temporal causation (max_lag=10)
2. **Information Bottleneck** — VAE compression + leave-one-out importance
3. **Wasserstein Drift** — Distribution shift monitoring for feature stability

---

## 14. Sentiment & News Analysis

### 14.1 OANDA Order Book Sentiment (`sentiment.py`)

```python
class SentimentAnalyzer:
    # Uses OANDA's public order/position book
    # Long/short ratios → contrarian signals
    # Order clusters near S/R levels
    # Cache: 30 minutes
```

### 14.2 FinBERT NLP Sentiment (`sentiment_engine.py`)

```python
class SentimentEngine:
    # Primary: ProsusAI/finbert (HuggingFace, free)
    # Fallback: VADER sentiment (no GPU needed)
    # Score: -1.0 (bearish) to +1.0 (bullish)
    # Cache: 500 entries
```

### 14.3 Live News Feed (`news_feed.py`)

**Sources (all free):**
- RSS: Reuters, CNBC, MarketWatch, Bloomberg, FT
- NewsAPI.org (free tier: 100 req/day)
- Finnhub (free tier)

**Pipeline:**
```
RSS/API → Filter by instrument keywords → FinBERT sentiment score → Aggregate per instrument → Trading signal bias
```

### 14.4 Economic Calendar (`news_filter.py`)

```python
class NewsFilter:
    # Source: ForexFactory API
    # High-impact events → block trading ±30 min
    # Medium-impact → warning only
    # Cache: 4 hours
```

---

## 15. Market Data Enrichment

### 15.1 External Data Sources (`market_data_enricher.py`)

| Source | Data | Update Frequency |
|---|---|---|
| yfinance | VIX, DXY, S&P500, bonds, sector ETFs | 4 hours |
| FRED API | Fed Funds Rate, CPI, Unemployment, GDP, Yield Curve | 4 hours |
| Alpha Vantage | Commodity prices, forex rates | 4 hours |

### 15.2 Generated Features

```
VIX level, VIX percentile, VIX regime (low/normal/high/extreme)
DXY direction, DXY momentum
S&P500 trend, SPX-forex correlation
Yield curve slope (10Y - 2Y)
Fed balance sheet trend
CPI year-over-year change
```

---

## 16. Multi-Timeframe Analysis

### 16.1 Classical MTF (`mtf_analyzer.py`)

```
H1 SuperTrend direction ──┐
H4 SuperTrend direction ──┼──▶ MTF Score
D1 SuperTrend direction ──┘

Score: 1.0 (all agree) / 0.5 (H4 agrees) / 0.0 (disagreement)
```

### 16.2 ML MTF (`mtf_lstm.py`)

```
M5  data → LSTM encoder (64h) ──┐
M15 data → LSTM encoder (64h) ──┤
H1  data → LSTM encoder (64h) ──┼──▶ Cross-Temporal Attention (4-head)
H4  data → LSTM encoder (64h) ──┘         │
                                           ▼
                                  Learnable importance weights
                                  (softmax over timeframes)
```

---

## 17. Portfolio Optimization

### 17.1 Methods (`portfolio_optimizer.py`)

**Markowitz Mean-Variance:**

```
max   w'μ - λ/2 × w'Σw
s.t.  Σw_i = 1, w_i ∈ [0.05, 0.40]
```

**Risk Parity:**

```
w_i ∝ 1/σ_i  (inverse volatility weighting)
Goal: equalize risk contribution from each asset
```

**Minimum Variance:**

```
min   w'Σw
s.t.  Σw_i = 1, w_i ∈ [0.05, 0.40]
```

**Maximum Sharpe Ratio:**

```
max   (w'μ - r_f) / √(w'Σw)
s.t.  Σw_i = 1, w_i ∈ [0.05, 0.40]
```

### 17.2 Regime Adjustment

| Regime | Covariance Σ | Expected Returns μ |
|---|---|---|
| VOLATILE | Σ × 1.5 | μ × 0.7 |
| TRENDING | unchanged | μ × 1.2 |
| NORMAL | unchanged | unchanged |

### 17.3 Efficient Frontier

Computed via constrained optimization at N target return levels using scipy SLSQP.

---

## 18. Backtesting Engine

### 18.1 Professional Metrics (`backtester.py`)

| Metric | Formula |
|---|---|
| Sharpe Ratio | `(E[R] - Rf) / σ(R) × √252` |
| Sortino Ratio | `(E[R] - Rf) / σ_downside × √252` |
| Calmar Ratio | `Annual Return / Max Drawdown` |
| Max Drawdown | `max((Peak - Trough) / Peak)` |
| Profit Factor | `Σ(wins) / |Σ(losses)|` |
| Expectancy | `WR × Avg_Win - LR × Avg_Loss` |
| Recovery Factor | `Net Profit / Max Drawdown` |
| Payoff Ratio | `Avg_Win / Avg_Loss` |

### 18.2 Monte Carlo Simulation

```python
def monte_carlo(trades, n_simulations=1000):
    # Resample trades with replacement
    # Generate 1000 equity curves
    # Calculate confidence intervals:
    #   P5  = 5th percentile (worst case)
    #   P50 = median (expected)
    #   P95 = 95th percentile (best case)
```

### 18.3 Stress Testing

```python
def stress_test(candles, vol_multipliers=[1.5, 2.0, 3.0]):
    # Amplify volatility by multiplier
    # Re-run backtest under stressed conditions
    # Compare: normal vs stressed metrics
```

---

## 19. Optimization & Training Pipeline

### 19.1 Continuous Learning (`training_manager.py`)

```
Schedule:
  Nightly (23:00 UTC-4): LSTM retraining with latest data
  Weekly (Friday 17:00): Full review + hyperparameter optimization

Nightly Pipeline:
  1. Collect last 24h candle data
  2. Feature engineering
  3. LSTM retraining (with data augmentation)
  4. MC Dropout validation
  5. Fisher Information change detection
  6. Wasserstein drift monitoring
  7. Compare accuracy: new vs previous
  8. Auto-revert if accuracy drops > 5%
  9. Log experiment to tracker

Weekly Pipeline:
  1. Walk-forward validation (5 folds)
  2. Granger causality feature selection
  3. HMM regime re-estimation
  4. Ensemble weight recalibration (EXP3)
  5. Hyperparameter optimization (Optuna, 50 trials)
  6. Performance report generation
```

### 19.2 Hyperparameter Optimization (`hyperopt.py`)

**Engine:** Optuna TPE (Tree-structured Parzen Estimator) with random search fallback.

**Search Spaces:**

```python
LSTM_PARAMS = {
    "hidden_size": [64, 128, 256],
    "num_layers": [1, 2, 3],
    "learning_rate": [1e-4, 1e-2],  # log-uniform
    "dropout": [0.1, 0.5],
    "sequence_length": [15, 60],
}

ENSEMBLE_PARAMS = {
    "lr_weight": [0.1, 0.5],
    "xgb_weight": [0.2, 0.6],
    "rl_weight": [0.1, 0.4],
    "agreement_threshold": [0.4, 0.8],
}

RISK_PARAMS = {
    "kelly_fraction": [0.25, 0.75],
    "max_size_pct": [1.0, 5.0],
    "mc_dropout_threshold": [0.1, 0.3],
    "wasserstein_alert": [0.3, 0.8],
}
```

**Objective:** Maximize Sharpe Ratio on last 50 trades.

---

## 20. Experiment Tracking

### 20.1 System (`experiment_tracker.py`)

```python
class ExperimentTracker:
    def log_experiment(params, metrics, model_hash, tags, notes) -> exp_id
    def log_training_run(trader) -> exp_id           # Auto-capture
    def get_best_experiment(metric, top_n) -> List
    def compare(exp_id_1, exp_id_2) -> Dict           # Side-by-side
    def rollback(trader, exp_id) -> Dict               # Restore params
    def get_leaderboard(metric, top_n) -> List
```

**Stored per experiment:**
- Timestamp, experiment ID
- All hyperparameters (ensemble weights, LSTM config, Kelly fraction, thresholds)
- Performance metrics (Sharpe, PF, win rate, net PnL)
- Model hash (MD5 of params)
- Tags and notes

**Storage:** JSON files in `experiments/` directory with `leaderboard.json` auto-updated.

---

## 21. Smart Alerts System

### 21.1 Alert Types (`smart_alerts.py`)

| Alert Type | Trigger | Default Cooldown | Severity |
|---|---|---|---|
| VAE Anomaly | Reconstruction error > threshold | 60 min | WARNING/CRITICAL |
| Wasserstein Drift | Distribution shift detected | 60 min | WARNING/CRITICAL |
| Fisher Regime | Fisher Information spike > 2σ | 120 min | WARNING/CRITICAL |
| HMM Transition | Regime state change | 30 min | INFO |
| MC Dropout | High uncertainty (σ > threshold) | 15 min | WARNING |
| Granger Shift | Causal structure change | 1440 min | INFO |
| Wavelet Energy | High noise level (>50%) | 60 min | WARNING |
| Drawdown | Equity drawdown > 3% | 30 min | WARNING/CRITICAL |
| Consecutive Losses | 4+ losses in a row | 30 min | WARNING/CRITICAL |
| Backtest Degradation | OOS Sharpe < 50% of IS | 1440 min | CRITICAL |

### 21.2 Anti-Spam

- Per-type cooldowns (configurable)
- CRITICAL alerts reduce cooldown to 1/3
- Max 50 alerts per day
- Delivery via Telegram

---

## 22. Dashboard & Monitoring

### 22.1 Flask API (`dashboard.py`) — 35+ Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/status` | Overall bot status |
| `GET /api/regime` | Current market regime |
| `GET /api/ml_stats` | ML model statistics |
| `GET /api/rl_stats` | DQN RL performance |
| `GET /api/streaming` | OANDA stream status |
| `GET /api/advisor` | FinBot advisor status |
| `GET /api/risk` | Risk management metrics |
| `GET /api/active_trades` | Open positions |
| `GET /api/equity_curve` | Equity history |
| `GET /api/performance` | Win rate, PF, Sharpe |
| `GET /api/signals` | Recent signals log |
| `GET /api/trades_history` | Closed trades |
| `GET /api/gpu_status` | GPU temp, VRAM, utilization |
| `GET /api/market_enrichment` | External data status |
| `GET /api/news_sentiment` | News feed + sentiment |
| `GET /api/advanced_learning` | 20 strategies status |
| `GET /api/training_status` | Training manager state |
| `GET /api/smart_alerts` | Recent alerts |
| `GET /api/kelly_status` | Kelly sizing metrics |
| `GET /api/walk_forward_status` | WF validation results |
| `GET /api/experiments` | Experiment leaderboard |
| `GET /api/portfolio_status` | Portfolio weights |
| `GET /api/v9_overview` | v9 modules overview |
| `GET /api/v10_overview` | v10 modules overview |
| `GET /dashboard` | Interactive HTML dashboard |

### 22.2 Interactive Dashboard (`dashboard_v2.html`)

6-tab Chart.js dashboard with auto-refresh (10s):

1. **Overview** — Equity curve, KPIs (balance, P&L, win rate, Sharpe), module status, recent signals
2. **Performance** — Monte Carlo distribution, drawdown chart, win/loss histogram
3. **ML & Strategies** — TF importance radar, wavelet energy decomposition, ensemble weights, Granger matrix
4. **Risk & Portfolio** — Kelly adjustments bar chart, portfolio weights donut, efficient frontier scatter
5. **Experiments** — Leaderboard table, walk-forward validation results, HyperOpt progress
6. **Alerts** — Alert timeline, summary statistics

---

## 23. Telegram Bot

### 23.1 Features (`telegram_bot.py`)

**Commands:**

| Command | Description |
|---|---|
| `/status` | Bot status + balance |
| `/trades` | Open positions |
| `/pnl` | Today's P&L |
| `/equity` | Equity curve (text) |
| `/risk` | Risk metrics |
| `/regime` | Current market regime |
| `/help` | Command list |

**Automated Alerts:**
- Trade entries and exits
- Drawdown warnings (3%, 5%)
- Consecutive loss alerts
- Regime transitions
- Daily/weekly summaries

### 23.2 Financial Advisor (`financial_advisor.py`)

Spanish-language AI advisor "FinBot" with:
- Contextual phrase generation
- Daily narrative summaries
- Performance commentary
- Risk alerts in natural language
- Trade recording and history

---

## 24. Docker Deployment

### 24.1 Container Stack

```yaml
# docker-compose.yml
services:
  bot:
    build: .
    image: ml-supertrend:v51
    runtime: nvidia                    # GPU passthrough
    ports: ["5000:5000"]               # Dashboard
    volumes:
      - ./data:/app/data               # Persistent data
      - ./models:/app/models           # Model weights
      - ./experiments:/app/experiments  # Experiment logs
      - ./.env:/app/.env:ro            # API keys
    command: python main.py --demo
```

### 24.2 Base Image

```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
# Includes: Python 3.11, PyTorch 2.4, CUDA 12.4, cuDNN 9
# Added: TA-Lib C library (compiled from source)
```

### 24.3 Quick Start

```bash
# Without Docker
pip install -r requirements_gpu.txt
python main.py --demo

# With Docker
docker compose up -d
docker compose logs -f bot

# Dashboard
open http://localhost:5000/dashboard
```

---

## 25. Configuration Reference

### 25.1 Environment Variables (`.env`)

```bash
# OANDA
OANDA_ACCOUNT_ID=xxx-xxx-xxxxxxx-xxx
OANDA_API_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxx
OANDA_DEMO_ACCOUNT_ID=xxx-xxx-xxxxxxx-xxx
OANDA_DEMO_API_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxx

# Telegram
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHI...
TELEGRAM_CHAT_ID=123456789

# External Data (free tiers)
FRED_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
ALPHA_VANTAGE_KEY=xxxxxxxxxxxxxxxx
NEWSAPI_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
FINNHUB_KEY=xxxxxxxxxxxxxxxxxxxxxxxx

# Crypto (optional)
BINANCE_API_KEY=xxx
BINANCE_API_SECRET=xxx
BYBIT_API_KEY=xxx
BYBIT_API_SECRET=xxx
```

### 25.2 Key Parameters (`config.py`)

```python
MAX_CONCURRENT_POSITIONS = 3
MAX_POSITIONS_PER_PAIR = 1
MAX_DAILY_TRADES = 10
MAX_DAILY_LOSS_PCT = 12.0
POLL_INTERVAL = 30  # seconds
USER_TZ_OFFSET = -4  # UTC-4

STRATEGY = StrategyParams(
    supertrend_factor=2.618,
    atr_period=10,
    adx_min=20,
    rsi_overbought=72,
    rsi_oversold=28,
    signal_threshold=0.35,
    confirmation_candles=1,
)
```

---

## 26. Mathematical Formulas Reference

### Position Sizing

```
Kelly: f* = (p×b - q) / b
Half-Kelly: f = f* / 2
Adjusted: f_adj = f × Π(multipliers)
Units = (Balance × f_adj) / SL_distance
```

### Risk Metrics

```
Sharpe = (E[R] - Rf) / σ(R) × √252
Sortino = (E[R] - Rf) / σ_down × √252
Calmar = Annual_Return / Max_DD
Profit_Factor = Σ(wins) / |Σ(losses)|
Expectancy = WR × Avg_Win - (1-WR) × Avg_Loss
```

### ML Models

```
LSTM: h_t = σ(W_h × [h_{t-1}, x_t] + b_h)
Attention: A = softmax(QK'/√d_k)V
DQN: Q(s,a) = r + γ × max Q(s',a')
VAE: L = ||x-x̂||² + β×KL(q(z|x)||p(z))
InfoNCE: L = -log(exp(sim(z_i,z_j)/τ) / Σ exp(sim(z_i,z_k)/τ))
Granger: F = ((RSS_r - RSS_u)/p) / (RSS_u/(n-2p-1))
Wasserstein: W₁ = (1/n) Σ |F⁻¹_P(i/n) - F⁻¹_Q(i/n)|
```

### Portfolio

```
Markowitz: max w'μ - λ/2 × w'Σw  s.t. Σw=1
Risk Parity: w_i ∝ 1/σ_i
Max Sharpe: max (w'μ - rf) / √(w'Σw)
Black-Litterman: E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹Π + P'Ω⁻¹Q]
```

---

## 27. Academic References

| # | Paper | Year | Strategy |
|---|---|---|---|
| 1 | Moody & Saffell — "Learning to Trade via Direct RL" | 2001 | Sharpe Loss |
| 2 | Sutton — "Learning to Predict by TD Methods" | 1988 | TD-λ |
| 3 | Hamilton — "A New Approach to Economic Analysis" | 1989 | HMM Regime |
| 4 | Koenker & Bassett — "Regression Quantiles" | 1978 | Quantile Regression |
| 5 | Auer et al. — "Non-stochastic Multi-armed Bandit" | 2002 | EXP3 |
| 6 | Um et al. — "Data Augmentation of Wearable Sensor Data" | 2017 | Augmentation |
| 7 | Ly et al. — "A Tutorial on Fisher Information" | 2017 | Fisher Information |
| 8 | Bengio et al. — "Curriculum Learning" | 2009 | Curriculum |
| 9 | Gal & Ghahramani — "Dropout as Bayesian Approximation" | 2016 | MC Dropout |
| 10 | Vaswani et al. — "Attention Is All You Need" | 2017 | Positional Encoding |
| 11 | Lim et al. — "Temporal Fusion Transformers" | 2021 | TFT/VSN |
| 12 | Vallender — "Calculation of Wasserstein Distance" | 1974 | Wasserstein |
| 13 | Yue et al. — "TS2Vec: Universal Time Series Representation" | 2022 | Contrastive |
| 14 | Hinton et al. — "Distilling Knowledge in Neural Nets" | 2015 | Distillation |
| 15 | Finn et al. — "Model-Agnostic Meta-Learning" | 2017 | MAML |
| 16 | Daubechies — "Ten Lectures on Wavelets" | 1992 | Wavelet |
| 17 | Kingma & Welling — "Auto-Encoding Variational Bayes" | 2014 | VAE |
| 18 | Velickovic et al. — "Graph Attention Networks" | 2018 | GNN |
| 19 | Tishby et al. — "The Information Bottleneck Method" | 2000 | Info Bottleneck |
| 20 | Granger — "Investigating Causal Relations by Econometric Models" | 1969 | Granger Causality |
| 21 | Markowitz — "Portfolio Selection" | 1952 | Portfolio Opt |
| 22 | Kelly — "A New Interpretation of Information Rate" | 1956 | Kelly Criterion |
| 23 | Black & Litterman — "Global Portfolio Optimization" | 1992 | Black-Litterman |
| 24 | Lopez de Prado — "Advances in Financial ML" | 2018 | Walk-Forward |
| 25 | Bailey & Lopez de Prado — "The Deflated Sharpe Ratio" | 2014 | Deflated Sharpe |

---

## 28. Installation & Setup

### 28.1 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores | i7-14700KF (20 cores) |
| RAM | 8 GB | 32 GB |
| GPU | — | RTX 4070 Super (12GB VRAM) |
| Storage | 5 GB | 50 GB SSD |

### 28.2 Software Requirements

```
Python 3.10+
PyTorch 2.4+ with CUDA 12.x (GPU)
NVIDIA Driver 550+
CUDA Toolkit 12.4
TA-Lib C library
```

### 28.3 Installation

```bash
# 1. Clone
git clone https://github.com/your-username/ML_SuperTrend_Bot.git
cd ML_SuperTrend_Bot

# 2. PyTorch (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Dependencies
pip install -r requirements_gpu.txt

# 4. TA-Lib (Linux)
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib
./configure --prefix=/usr && make && sudo make install
pip install ta-lib

# 5. Configure
cp .env.example .env
# Edit .env with your API keys

# 6. Run
python main.py --demo
```

### 28.4 Dependencies Summary

```
Core:        numpy, pandas, requests, flask, flask-cors
ML:          torch, xgboost, scikit-learn, scipy, hmmlearn
Wavelets:    PyWavelets
Optimization: optuna
NLP:         transformers, tokenizers, vaderSentiment
Data:        yfinance, fredapi
GPU:         pynvml, gputil
Charts:      plotly, matplotlib
Telegram:    python-telegram-bot
Utils:       python-dotenv, schedule, colorama
```

---

> **ML SuperTrend v51** — 29,100 lines | 59 modules | 20 scientific strategies | GPU-accelerated
> 
> Built for the RTX 4070 Super + i7-14700KF platform
