# ZoneX — ML-Driven Market State Engine for Crypto Futures

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io/)
[![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

**Institutional-grade quantitative research framework that replaces linear indicator scoring with zone-based market state classification and XGBoost probability modeling.**

---

## 🎯 The Problem

Traditional trading systems use **linear scoring**:
score = RSI_weight × RSI_value + volume_weight × volume_value + ...

text

This assumes every indicator works the same way in all conditions. **Real markets don't work like that.**

| RSI Zone | Actual Win Rate | Linear Model Says |
|----------|----------------|-------------------|
| RSI < 35 | 52-54% | "Oversold = bad" |
| RSI 45-55 | 42% | "Neutral = OK" |
| RSI > 65 on 1h | 57% | "Overbought = bad" |

The "comfort zone" (RSI 45-55) is actually the **worst zone** to trade. Linear models get this completely wrong.

---

## 🧠 The Solution: Zone-Based Market State Engine

ZoneX partitions every feature into **empirical profitability zones** — derived from historical data, not assumptions.
Instead of: RSI_score = RSI × 0.3
We use: IF RSI < 35 → score = +2.0 (WR=54%, PF=1.34)
IF RSI 45-55 → score = -0.8 (WR=42%, PF=0.46)
IF RSI > 65 AND timeframe=1h → score = +1.0 (WR=57%, PF=1.39)

text

Zones are **regime-conditional** and **timeframe-specific**. RSI works differently in chop vs trend_up. The 15m timeframe inverts compared to 1h.

---

## 📊 Research Backing

All zone tables are derived from rigorous statistical analysis:
Dataset: 300+ USDT perpetual futures pairs
Timeframes: 15m, 1h, 4h
Period: 90 trading days
Features: 33 per data point
XGBoost AUC: 0.765 (temporal validation, no look-ahead bias)

text

### Key Empirical Findings

| Finding | Evidence | Action |
|---------|----------|--------|
| RSI < 35 is the only consistently profitable zone | WR 52-54%, PF 1.32-1.34 | Max score |
| RSI 45-55 is a death zone | WR 42%, PF 0.70 | Penalize |
| Compression 50-70% destroys edge | WR 39% | Hard block |
| 15m timeframe: SHORT only | WR 58%, PF 1.69 | Direction filter |
| 1h timeframe: RSI > 65 is profitable | WR 57%, PF 1.39 | Zone bonus |
| Momentum > 5% is chasing | WR 31% | Hard block |
| Hours 17, 19, 20, 22 UTC | WR 17-29% | Session block |

---

## 🏗️ Architecture
Market Data (Binance/Bybit)
│
┌──────▼──────┐
│ BTC Regime │ ← trend_up/chop/volatile/trend_down
│ Detection │
└──────┬──────┘
│
┌──────▼──────┐
│ Multi-TF │ ← 15m / 1h / 4h
│ Scanner │
└──────┬──────┘
│
┌────────────┼────────────┐
│ │ │
┌────▼───┐ ┌─────▼──────┐ ┌─▼──────────┐
│ Zone │ │ Archetype │ │ Interaction │
│ Scoring │ │ Engine │ │ Layer │
└────┬───┘ └─────┬──────┘ └─┬──────────┘
│ │ │
└────────────┼────────────┘
│
┌──────▼──────┐
│ Hard │ ← RR/SL/TP/ATR/momentum filters
│ Filters │
└──────┬──────┘
│
┌──────▼──────┐
│ XGBoost │ ← Probability output
│ Ranker │
└──────┬──────┘
│
┌──────▼──────┐
│ Trade │
│ Decision │
└─────────────┘

text

---

## 📁 Project Structure
ZoneX/
├── pattern_lab/ # Pattern Detection R&D
│ ├── pattern_engine.py # Structural TA figure detector
│ │ # Head & Shoulders, Double Top/Bottom,
│ │ # Flag/Pennant, Wedges, Triangles
│ ├── calibrate.py # Grid search for optimal parameters
│ ├── build_ml_dataset_v2.py # Feature extraction pipeline (33 features)
│ ├── download_data.py # Multi-symbol OHLCV fetcher
│ └── visualize.py # Pattern visualization with mplfinance
│
├── research/ # Statistical Research Suite
│ ├── 01_nonlinear_zones.py # Zone-based profitability analysis
│ ├── 02_interactions.py # Feature interaction effects
│ ├── 03_regime_dependence.py # Regime-conditional performance
│ ├── 04_archetypes.py # Market state archetype analysis
│ ├── 05_timeframe_physics.py # Timeframe-specific dynamics
│ ├── 06_probability_calibration.py # Prediction vs actual outcome
│ ├── 07_stability.py # Cross-temporal robustness testing
│ ├── 08_feature_drift.py # Feature importance evolution
│ ├── 09_final_report.py # Executive summary generator
│ └── results/ # JSON research outputs
│
├── requirements.txt
├── LICENSE
└── README.md

text

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USER/ZoneX.git
cd ZoneX

# 2. Install
pip install -r requirements.txt

# 3. Download market data
cd pattern_lab
python download_data.py

# 4. Build ML dataset
python build_ml_dataset_v2.py

# 5. Run research pipeline
cd ../research
python 01_nonlinear_zones.py
python 02_interactions.py
# ... run all 01-09
python 09_final_report.py
🔬 Research Modules
#	Module	Question Answered
01	Nonlinear Zones	Which feature zones have positive expectancy?
02	Interactions	How do features modify each other's meaning?
03	Regime Dependence	Which features work in which BTC regime?
04	Archetypes	Do recurring market states predict outcomes?
05	Timeframe Physics	How does market behavior differ by timeframe?
06	Calibration	Do predicted probabilities match reality?
07	Stability	Do edges persist across months?
08	Feature Drift	How does feature importance evolve?
09	Final Report	Executive summary of all findings
🎓 Key Technical Concepts
Zone-Based Scoring
Traditional score = Σ(weight × indicator) assumes linear, monotonic relationships. Real markets are nonlinear. ZoneX uses lookup tables derived from empirical distributions.

Regime Conditioning
Features are conditioned on BTC regime. An edge in chop may be a trap in trend_up. All zone tables are regime-specific.

Confidence Weighting
Zones with small sample sizes (n < 300) have their scores scaled down by √(n/300). Prevents overfitting to noise.

Temporal Validation
All models use strict temporal ordering — trained on earlier periods, tested on later periods. Zero look-ahead bias.

Stability Over Optimization
An edge that shows 80% WR in one month but collapses the next is rejected. ZoneX prioritizes stable, modest edges over optimized, fragile ones.

📈 Sample Output
text
════════════════════════════════════════════════════════════
  ETHUSDT 15m  ↓ SHORT  [Флаг]
  State: regime=chop tf=15m
────────────────────────────────────────────────────────────
  ✅ Hard Filters: PASSED
────────────────────────────────────────────────────────────
  rsi             +2.5 × 3.0 =  +7.50  [15m] RSI 25-35 WR=65% PF=3.12 c=1.00
  momentum        +0.5 × 2.5 =  +1.25  mom -2..0% WR=50% PF=1.22 c=1.00
  atr             +0.7 × 1.5 =  +1.05  [chop] ATR 0.5-1 PF=1.32 c=1.00
  volume          +0.5 × 1.5 =  +0.75  [chop] vol 0.3-0.6 PF=1.26 c=1.00
  compression     +0.1 × 1.0 =  +0.10  [chop] comp 20-40 PF=1.05 c=1.00
  position        +1.0 × 2.0 =  +2.00  шорт у верха (78%)
  direction       +1.2 × 1.5 =  +1.80  short [15m]
────────────────────────────────────────────────────────────
  Interactions:   +1.50  15m+RSI<35+short=WR65%
  Regime base:    +0.50  chop
  Total score:    +16.45
  Probability:    0.999
  ✅ БРАТЬ
════════════════════════════════════════════════════════════
📦 Requirements
text
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
matplotlib>=3.7.0
mplfinance>=0.12.0
requests>=2.28.0
scikit-learn>=1.2.0
xgboost>=1.7.0
👤 Author
Vlada — Quantitative Researcher & Systematic Crypto Trader

⚠️ Disclaimer
This software is for research and educational purposes only. Past performance does not guarantee future results. Cryptocurrency trading involves substantial risk of loss. The author assumes no liability for any trading losses.

📝 License
MIT License — see LICENSE

If this project helps your research, please ⭐ star it!