# Polymarket Prediction System

A state-of-the-art machine learning system for predicting Polymarket outcomes using advanced ensemble methods.

## Overview

This project uses sophisticated ML algorithms including XGBoost, LightGBM, and stacking ensembles to analyze prediction markets on Polymarket and generate trading signals with confidence scores.

## Features

- **Advanced ML Models**: Combines XGBoost and LightGBM with stacking ensembles
- **Real-time Market Analysis**: Fetches live trading data from Polymarket API
- **Quantitative Metrics**: RSI, volatility, order book imbalance, expected value calculations
- **Risk Management**: Kelly criterion position sizing and terminal risk adjustments
- **Smart Filtering**: Automatically excludes resolved and low-volume markets

## Components

- `main.py` - Main prediction engine and CLI interface
- `polymarket_fetcher.py` - Polymarket API integration for fetching market data
- `prediction_model.py` - ML models and prediction algorithms
- `test_real_data.py` - Testing with real market data
- `test_real_training.py` - Model training and validation tests

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/polymarket-prediction-system.git
cd polymarket-prediction-system

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run predictions on top markets:

```bash
# Analyze top 10 markets
python main.py

# Analyze custom number of markets
python main.py 20
```

## Training on resolved snapshots

```bash
python -m data.build_snapshots --out data/features/snapshots.parquet
python train_snapshot_model.py --snapshots_path data/features/snapshots.parquet
```

Sentiment features are optional; toggle providers and keys in `config/sentiment_config.yaml`. When providers are unavailable, the snapshot builder and live inference fill the canonical sentiment columns with `NaN` so feature alignment is preserved.

## Backtest

Replay snapshot-based trades with the probability bot rules in paper mode:

```bash
python -m backtest.run \
  --snapshots data/features/snapshots.parquet \
  --model models/latest.joblib \
  --out results/backtest/ \
  --slippage_bps 5
```

The runner replays snapshot rows chronologically, applies the same threshold rules and risk manager as the live bot, and saves an equity curve, trade log, and summary metrics (JSON) under the output directory.

## Monitoring & Metrics

- Training, backtests, and the live bot emit JSONL metrics to a run directory (defaults under `results/run_<timestamp>/metrics.jsonl`).
- Event types include `model_eval`, `decision`, `trade`, `sentiment`, `health` (throttles/backoff/staleness), and `filter_skip`.
- Summaries and CSV exports can be generated via:

```bash
python -m metrics.summarize --run_dir results/run_20240101_120000
```

which writes `summary.json`, equity/drawdown CSVs, and counts for skips/sentiment/decisions.

## Market quality filters

Trading, backtesting, and NegRisk arbitration all call a shared market-quality filter (`bot/market_filters.py`) configured via `config/trading_bot_config.yaml`. Filters can exclude markets with:

- Volume below `min_24h_volume`
- Spreads above `max_spread_abs` or `max_spread_pct`
- Top-of-book depth below `min_top_of_book_depth`
- Fewer than `min_trades_last_24h` recent trades (when available)
- Ambiguous rules/description text (length + keyword heuristic)
- Time-to-resolution below `skip_if_time_to_resolve_lt`

Lowering thresholds increases coverage but risks stale/illiquid fills; tightening them improves execution quality at the cost of fewer opportunities. Skip reasons are logged to aid tuning.

## Output

The system provides:
- Current vs. predicted prices
- Buy/Sell/Hold recommendations
- Confidence scores
- Trading signals (STRONG BUY, BUY, HOLD)
- Key insights (RSI, volatility, order book pressure)
- Kelly-optimized position sizes

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## Disclaimer

This software is for educational and research purposes only. It is NOT financial advice. Prediction markets involve substantial risk. Always do your own research and never invest more than you can afford to lose.

## License

MIT License - See LICENSE file for details
