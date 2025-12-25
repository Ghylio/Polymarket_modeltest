#!/usr/bin/env python3
"""Live trading bot entrypoint using probability model outputs."""

import argparse
import signal
import sys
import time
from typing import List

from trading_bot import ProbabilityTradingBot


def _format_trade(trade: dict) -> str:
    return (
        f"{trade['market_id']} | {trade['side']} @ {trade['price']:.3f} | "
        f"p_mean={trade['p_mean']:.3f} (lcb={trade['p_lcb']:.3f}, ucb={trade['p_ucb']:.3f}) "
        f"bid={trade.get('bid', float('nan')):.3f} ask={trade.get('ask', float('nan')):.3f}"
    )


def run_bot(loop: bool, interval: int, limit: int, threshold: float, paper: bool, models_dir: str):
    bot = ProbabilityTradingBot(models_dir=models_dir, threshold=threshold, paper_trading=paper)

    def _shutdown(*_):
        bot.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        trades = bot.run_once(limit=limit)
        if trades:
            print(f"Executed {len(trades)} simulated trades:")
            for t in trades:
                print("  " + _format_trade(t))
        else:
            print("No qualifying trades this cycle")

        if not loop:
            break
        time.sleep(interval)


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Polymarket probability trading bot")
    parser.add_argument("--limit", type=int, default=10, help="Number of markets per poll")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between polls")
    parser.add_argument("--threshold", type=float, default=0.05, help="Edge threshold vs price")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--paper", action="store_true", help="Enable paper trading mode")
    parser.add_argument("--models-dir", default="models", help="Path to trained model artifacts")

    args = parser.parse_args(argv)
    run_bot(loop=args.loop, interval=args.interval, limit=args.limit, threshold=args.threshold, paper=args.paper, models_dir=args.models_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
