"""Summarize metrics events written by training/backtest/live runs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def _load_events(path: Path) -> List[Dict]:
    events: List[Dict] = []
    if not path.exists():
        return events
    with path.open() as fh:
        for line in fh:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _equity_curve(events: Iterable[Dict]) -> pd.DataFrame:
    rows = [e for e in events if e.get("type") == "health" and "equity" in e]
    return pd.DataFrame(rows)


def _drawdown(equity_df: pd.DataFrame) -> pd.DataFrame:
    if equity_df.empty or "equity" not in equity_df.columns:
        return pd.DataFrame()
    equity_df = equity_df.sort_values("timestamp")
    equity = equity_df["equity"].astype(float)
    cummax = equity.cummax()
    dd = equity - cummax
    return pd.DataFrame({"timestamp": equity_df["timestamp"], "drawdown": dd})


def summarize_run(run_dir: Path) -> Dict:
    metrics_path = Path(run_dir) / "metrics.jsonl"
    events = _load_events(metrics_path)

    summary: Dict = {
        "decision_counts": Counter(),
        "filter_skip_counts": Counter(),
        "sentiment_events": 0,
        "sentiment_nan_pct": [],
        "trade_pnl": [],
    }

    decisions: List[Dict] = []
    trades: List[Dict] = []
    sentiments: List[Dict] = []

    for e in events:
        etype = e.get("type")
        if etype == "decision":
            action = e.get("action") or "unknown"
            summary["decision_counts"][action] += 1
            decisions.append(e)
        elif etype == "filter_skip":
            for reason in e.get("reasons", []):
                summary["filter_skip_counts"][reason] += 1
        elif etype == "sentiment":
            summary["sentiment_events"] += 1
            if "percent_nan" in e:
                summary["sentiment_nan_pct"].append(float(e["percent_nan"]))
            sentiments.append(e)
        elif etype == "trade":
            if "pnl" in e:
                summary["trade_pnl"].append(float(e["pnl"]))
            trades.append(e)

    equity_df = _equity_curve(events)
    drawdown_df = _drawdown(equity_df)

    summary["avg_sentiment_nan"] = float(pd.Series(summary["sentiment_nan_pct"]).mean()) if summary["sentiment_nan_pct"] else None
    summary["total_pnl"] = float(sum(summary["trade_pnl"])) if summary["trade_pnl"] else 0.0

    # Persist outputs
    serializable_summary = summary.copy()
    serializable_summary["decision_counts"] = dict(summary["decision_counts"])
    serializable_summary["filter_skip_counts"] = dict(summary["filter_skip_counts"])

    summary_path = Path(run_dir) / "summary.json"
    summary_path.write_text(json.dumps(serializable_summary, indent=2))

    if not equity_df.empty:
        equity_df.to_csv(Path(run_dir) / "equity.csv", index=False)
    if not drawdown_df.empty:
        drawdown_df.to_csv(Path(run_dir) / "drawdown.csv", index=False)

    if summary["filter_skip_counts"]:
        skip_df = pd.DataFrame(
            {"reason": list(summary["filter_skip_counts"].keys()), "count": list(summary["filter_skip_counts"].values())}
        )
        skip_df.to_csv(Path(run_dir) / "skip_counts.csv", index=False)

    if decisions:
        pd.DataFrame(decisions).to_csv(Path(run_dir) / "decisions.csv", index=False)
    if sentiments:
        pd.DataFrame(sentiments).to_csv(Path(run_dir) / "sentiment.csv", index=False)

    return summary


def main(argv=None):  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Summarize metrics run directory")
    parser.add_argument("--run_dir", required=True, help="Path to run directory containing metrics.jsonl")
    args = parser.parse_args(argv)
    summary = summarize_run(Path(args.run_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
