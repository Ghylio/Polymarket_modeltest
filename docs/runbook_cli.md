# CLI Runbook

This runbook summarizes all discovered command-line interfaces, their options, configuration files, and environment variables.

## Trading bot
- Command: `python main.py`
- Options (all optional):
  - `--limit` *(int, default=10)* – markets per poll.
  - `--interval` *(int, default=30)* – seconds between polls.
  - `--threshold` *(float, default=0.05)* – edge threshold versus price.
  - `--loop` *(flag)* – run continuously.
  - `--paper` *(flag)* – enable paper trading mode.
  - `--models-dir` *(str, default="models")* – path to trained model artifacts.
- Config / env:
  - Uses trading bot YAML defaults via `trading_bot.load_trading_bot_config` (`config/trading_bot_config.yaml`).
  - Respects `POLYMARKET_MODELS` for model directory; `SENTIMENT_DB` and `RESEARCH_DB` override local stores.
- Run locally: `python main.py --loop --interval 60 --threshold 0.05 --models-dir models`

## Sentiment ingestion service
- Command: `python -m sentiment.ingest`
- Options:
  - `--db` *(Path, default `data/sentiment.db`)* – SQLite path.
  - `--config` *(Path, default `config/sentiment_config.yaml` if omitted)*.
  - `--once` *(flag)* – run a single iteration.
- Config: sentiment provider YAML `config/sentiment_config.yaml` loaded by `data.sentiment_config.load_sentiment_config`.
- Run locally: `python -m sentiment.ingest --db data/sentiment.db --config config/sentiment_config.yaml --once`

## Research ingestion service
- Command: `python -m research.ingest` (legacy: `python research/ingest_service.py` still works)
- Options:
  - `--db` *(Path, default `data/sentiment.db`)* – SQLite path used for both research features and sentiment docs.
  - `--interval_sec` *(int, default 300)* – polling interval.
  - `--max_markets` *(int, default 50)* – markets per cycle.
  - `--config` *(Path, default `config/research_config.yaml`)* – research settings.
- Config: `config/research_config.yaml` parsed by `load_research_config` inside `research.ingest_service`.
- Env: expects `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`, `OPENAI_ORG_ID`) for OpenAI client in `research.llm_openai`.
- Run locally: `python -m research.ingest --db data/sentiment.db --config config/research_config.yaml --interval_sec 300 --max_markets 50`

## Snapshot dataset builder
- Command: `python -m data.build_snapshots` (legacy: `python data/build_snapshots.py` still works)
- Options:
  - `--out` *(required str)* – output Parquet path.
  - `--limit` *(int, default=50)* – number of markets.
  - `--min-volume` *(float, default=0)* – minimum 24h volume filter.
  - `--fidelity` *(int, default=60)* – minute fidelity for price history.
  - `--sentiment-db` *(Path, default `data/sentiment.db`)* – sentiment SQLite path.
  - `--allow_online_sentiment_fetch` *(flag)* – backfill via providers when local store missing.
  - `--research-db` *(Path, default `data/sentiment.db`)* – research SQLite path.
  - `--use-research` *(flag)* – include research features.
- Run locally: `python -m data.build_snapshots --out data/features/snapshots.parquet --limit 100 --sentiment-db data/sentiment.db --use-research --research-db data/sentiment.db`

## Snapshot model trainer
- Command: `python train_snapshot_model.py`
- Options:
  - `--snapshots_path` *(str, default `data/features/snapshots.parquet`)* – snapshot dataset path.
  - `--out` *(str, default `models/snapshot_model.joblib`)* – trained model output.
  - `--legacy-directional` *(flag)* – train on price direction instead of resolution labels.
  - `--run_dir` *(str, optional)* – metrics directory (auto-created if omitted).
- Run locally: `python train_snapshot_model.py --snapshots_path data/features/snapshots.parquet --out models/snapshot_model.joblib`

## Backtester
- Command: `python -m backtest.run` (legacy: `python backtest/run.py` still works)
- Options (required unless noted):
  - `--snapshots` *(required str)* – snapshots parquet path.
  - `--model` *(required str)* – trained model artifact.
  - `--out` *(required str)* – results directory.
  - `--threshold` *(float, default=0.05)* – decision threshold.
  - `--slippage_bps` *(float, default=5.0)*.
  - `--slippage_ticks` *(float, default=0.0)*.
  - `--size` *(float, default=1.0)* – position size multiplier.
  - `--disable_gates` *(flag)* – disable evaluation throttles.
- Run locally: `python -m backtest.run --snapshots data/features/snapshots.parquet --model models/snapshot_model.joblib --out results/backtest`

## Final report generator
- Command: `python final_report.py`
- Options: none; prints a progress-style report.

## Metrics summarizer
- Command: `python -m metrics.summarize` (legacy: `python metrics/summarize.py` still works)
- Options:
  - `--run_dir` *(required str)* – directory containing `metrics.jsonl`.
- Run locally: `python -m metrics.summarize --run_dir results/backtest`

## Configuration files
- YAML files:
  - `config/trading_bot_config.yaml`
  - `config/subgraph_config.yaml`
  - `config/sentiment_config.yaml`
  - `config/research_config.yaml`
- Config loaders:
  - Trading bot: `trading_bot.load_trading_bot_config`.
  - Subgraph: `data.subgraph_config.load_subgraph_config`.
  - Sentiment: `data.sentiment_config.load_sentiment_config`.
  - Research: `research.ingest_service.load_research_config`.

## Environment variables
- `POLYMARKET_MODELS`: overrides model directory for trading bot.
- `SENTIMENT_DB` / `RESEARCH_DB`: override SQLite paths for stores.
- `OPENAI_API_KEY` (plus optional `OPENAI_BASE_URL`, `OPENAI_ORG_ID`): required for research ingestion LLM calls.
- `POLYMARKET_L1_PRIVATE_KEY`, `POLYMARKET_L2_API_KEY`, `POLYMARKET_L2_API_SECRET`: canonical CLOB credentials consumed by `clob_client.ClobAuth.from_env` (other prefixes are only honored if explicitly passed to `from_env`).

## Help command outputs
- `python main.py --help` – shows trading bot flags.
- `python -m sentiment.ingest --help` – shows sentiment ingestion flags.
- `python -m research.ingest --help` – research ingestion flags (same as legacy `python research/ingest_service.py --help`).
