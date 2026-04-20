# ML SuperTrend v51 — Algorithmic Trading Bot

> **GPU-Optimized Trading Bot with 20 Scientific ML Strategies**

## Overview

ML SuperTrend v51 is a production-grade algorithmic trading bot combining classical technical analysis (SuperTrend, RSI, ADX, MACD) with 20 cutting-edge machine learning strategies.

- **29,100+ lines** of Python across **59 modules**
- **20 scientific strategies** backed by academic papers
- **Multi-broker**: OANDA v20, Binance Futures, Bybit V5
- **GPU-accelerated**: PyTorch CUDA 12.x (RTX 4070 Super)
- **Real-time streaming**: OANDA v20 chunked HTTP
- **Continuous learning**: Nightly retraining + weekly review

## Quick Start

```bash
# Install (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_gpu.txt

# Run
python main.py --demo

# Dashboard
open http://localhost:5000/dashboard
```

## Docker

```bash
docker compose up -d
```

## 20 Scientific Strategies

| # | Strategy | Paper |
|---|---|---|
| 1 | Sharpe-Aware Loss | Moody & Saffell (2001) |
| 2 | TD-Lambda Learning | Sutton (1988) |
| 3 | HMM Regime Detection | Hamilton (1989) |
| 4 | Quantile Regression | Koenker & Bassett (1978) |
| 5 | EXP3 Online Learning | Auer et al. (2002) |
| 6 | Data Augmentation | Um et al. (2017) |
| 7 | Fisher Information | Ly et al. (2017) |
| 8 | Curriculum Learning | Bengio et al. (2009) |
| 9 | MC Dropout (Bayesian) | Gal & Ghahramani (2016) |
| 10 | Financial Positional Encoding | Vaswani et al. (2017) |
| 11 | Temporal Fusion Transformer | Lim et al. (2021) |
| 12 | Wasserstein Distance | Vallender (1974) |
| 13 | Contrastive Learning | Yue et al. (2022) |
| 14 | Knowledge Distillation | Hinton et al. (2015) |
| 15 | MAML Meta-Learning | Finn et al. (2017) |
| 16 | Wavelet Decomposition | Daubechies (1992) |
| 17 | VAE Anomaly Detection | Kingma & Welling (2014) |
| 18 | Graph Neural Network | Velickovic et al. (2018) |
| 19 | Information Bottleneck | Tishby et al. (2000) |
| 20 | Granger Causality | Granger (1969) |

## Architecture

See [DOCUMENTATION.md](DOCUMENTATION.md) for complete technical documentation including architecture diagrams, mathematical formulas, API reference, and deployment guide.

## Hardware

Optimized for: **i7-14700KF + RTX 4070 Super (12GB VRAM)**

## License

Private / All Rights Reserved
