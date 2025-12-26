"""
Snapshot dataset builder for probability modeling.

This module constructs labeled snapshots for resolved markets using only
information available at each snapshot time. Snapshots are bucketed by the
remaining time to resolution and include existing feature engineering output
from the live predictor to keep feature parity across training and inference.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.sentiment_config import load_sentiment_config
from data.sentiment_features import (
    SENTIMENT_COLUMNS,
    SENTIMENT_WINDOWS,
    SentimentFeatureBuilder,
    SentimentScorer,
    canonicalize_sentiment_features,
)
from data.sentiment_providers import build_providers_from_config
from data.subgraph_client import SubgraphClient
from data.subgraph_config import load_subgraph_config
from data.subgraph_features import SubgraphFeatureJoiner
from polymarket_fetcher import PolymarketFetcher
from prediction_model import MarketFeatureExtractor
from sentiment.store import DocumentStore, aggregate_scores

try:  # Parquet dependency is optional at runtime
    import pyarrow  # noqa: F401

    HAS_PARQUET = True
except ImportError:  # pragma: no cover - exercised in environments without pyarrow
    HAS_PARQUET = False


@dataclass
class SnapshotBuilder:
    """Build supervised snapshots for resolved markets."""

    fetcher: Optional[PolymarketFetcher] = None
    buckets: List[timedelta] = field(
        default_factory=lambda: [
            timedelta(days=30),
            timedelta(days=14),
            timedelta(days=7),
            timedelta(days=3),
            timedelta(hours=24),
            timedelta(hours=6),
            timedelta(hours=1),
        ]
    )
    schema_version: str = "snapshot_v1"
    use_sentiment: bool = True
    sentiment_db_path: Path = Path("data/sentiment.db")
    allow_online_sentiment_fetch: bool = False
    sentiment_store: Optional[DocumentStore] = None
    use_subgraph: bool = False
    subgraph_client: Optional[SubgraphClient] = None
    subgraph_joiner: Optional[SubgraphFeatureJoiner] = None

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
        self.fetcher = self.fetcher or PolymarketFetcher(verbose=False)
        self.feature_extractor = MarketFeatureExtractor()
        cfg = load_sentiment_config()
        self.use_sentiment = bool(cfg.get("sentiment", {}).get("enabled", False)) and self.use_sentiment
        self.sentiment_coverage: Tuple[int, int] = (0, 0)  # (windows_with_docs, total_windows)
        self.sentiment_store = self.sentiment_store or (
            DocumentStore(self.sentiment_db_path) if self.use_sentiment else None
        )
        self.allow_online_sentiment_fetch = bool(self.allow_online_sentiment_fetch)

        if self.use_sentiment:
            providers = build_providers_from_config(cfg)
            scorer = SentimentScorer(model_name=cfg.get("sentiment", {}).get("model"))
            self.sentiment_builder = SentimentFeatureBuilder(
                providers=providers, scorer=scorer, enabled=bool(providers)
            )
        else:
            self.sentiment_builder = None

        sg_cfg = load_subgraph_config()
        self.use_subgraph = self.use_subgraph or bool(
            sg_cfg.get("subgraph", {}).get("enabled", False)
        )
        if self.use_subgraph:
            volume_lookback = sg_cfg.get("subgraph", {}).get("volume_lookback_hours", 24)
            self.subgraph_client = self.subgraph_client or SubgraphClient(
                subgraph_url=sg_cfg.get("subgraph", {}).get("url"),
                resolution_url=sg_cfg.get("subgraph", {}).get("resolution_url"),
            )
            self.subgraph_joiner = SubgraphFeatureJoiner(
                client=self.subgraph_client,
                volume_lookback_hours=int(volume_lookback),
                enabled=True,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_snapshots(
        self,
        markets: Iterable[Dict],
        trades_by_market: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Create snapshot dataframe from resolved markets and trade history."""

        rows: List[Dict] = []
        for market in markets:
            market_id = market.get("id") or market.get("conditionId")
            if not market_id:
                continue

            try:
                resolution_ts = self._resolution_timestamp(market)
                label = self._resolution_label(market)
            except (KeyError, ValueError, TypeError):
                # Skip markets without clear resolution metadata
                continue

            trades_df = trades_by_market.get(market_id)
            if trades_df is None or trades_df.empty:
                continue

            prepared_trades = self._prepare_trades(trades_df)
            if prepared_trades.empty:
                continue

            rows.extend(
                self._snapshots_for_market(
                    market=market,
                    market_id=market_id,
                    resolution_ts=resolution_ts,
                    trades_df=prepared_trades,
                    label=label,
                )
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        if self.use_sentiment and self.sentiment_coverage[1] > 0:
            with_docs, total = self.sentiment_coverage
            coverage_pct = 100.0 * with_docs / float(total)
            self.logger.info(
                "Sentiment coverage: %.1f%% windows had docs (\%s/%s)",
                coverage_pct,
                with_docs,
                total,
            )

        if self.subgraph_joiner:
            df = self.subgraph_joiner.enrich_snapshots(df)
        return df.sort_values(["market_id", "snapshot_ts"]).reset_index(drop=True)

    def fetch_and_build(
        self, limit: int = 50, min_volume: float = 0, fidelity: int = 60
    ) -> pd.DataFrame:
        """Fetch resolved markets and trades, then build snapshots."""

        markets = self._fetch_resolved_markets(limit=limit, min_volume=min_volume)
        trades_by_market: Dict[str, pd.DataFrame] = {}

        for market in markets:
            market_id = market.get("id")
            yes_token, _ = self.fetcher.get_token_ids_for_market(market)
            if not yes_token:
                continue

            prices = self.fetcher.get_prices_history(
                yes_token, interval="max", fidelity=fidelity
            )
            if prices.empty:
                continue

            prices = prices.rename(columns={"price": "price", "timestamp": "timestamp"})
            prices["size"] = 1.0  # CLOB history lacks sizes; assign unit volume
            trades_by_market[market_id] = prices[["timestamp", "price", "size"]]

        return self.build_snapshots(markets, trades_by_market)

    def save(self, dataset: pd.DataFrame, out_path: Path) -> Path:
        """Persist dataset as Parquet with versioned schema."""

        if dataset.empty:
            raise ValueError("Snapshot dataset is empty; nothing to save.")

        if not HAS_PARQUET:
            raise ImportError(
                "pyarrow is required to write Parquet files. Install with `pip install pyarrow`."
            )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(out_path, index=False)
        return out_path

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------
    def _snapshots_for_market(
        self,
        market: Dict,
        market_id: str,
        resolution_ts: pd.Timestamp,
        trades_df: pd.DataFrame,
        label: int,
    ) -> List[Dict]:
        rows: List[Dict] = []

        for bucket in self.buckets:
            snapshot_ts = resolution_ts - bucket
            truncated = trades_df[trades_df["timestamp"] <= snapshot_ts]
            if truncated.empty:
                continue

            latest_row = truncated.iloc[-1]
            bid = latest_row["bid"] if "bid" in truncated.columns else np.nan
            ask = latest_row["ask"] if "ask" in truncated.columns else np.nan
            p_mid = self._midpoint(latest_row, bid=bid, ask=ask)
            spread = self._spread(bid, ask, fallback=market.get("spread"))

            trade_features = self.feature_extractor.extract_trade_features(truncated)
            market_features = self._market_features_at_snapshot(
                market=market,
                price_hint=p_mid,
            )
            feature_vector = self.feature_extractor.combine_features(
                trade_features, market_features
            ).flatten()
            feature_names = self.feature_extractor.get_feature_names()
            feature_dict = dict(zip(feature_names, feature_vector.tolist()))

            sentiment_features: Dict[str, float] = {}
            if self.use_sentiment:
                sentiment_features = self._sentiment_features_from_store(
                    market_id=market_id, market=market, snapshot_ts=pd.to_datetime(snapshot_ts)
                )
            else:
                sentiment_features = canonicalize_sentiment_features({})

            rows.append(
                {
                    "schema_version": self.schema_version,
                    "market_id": market_id,
                    "event_id": market.get("eventId")
                    or market.get("event_id")
                    or market_id,
                    "snapshot_ts": snapshot_ts,
                    "time_to_resolve_hours": bucket.total_seconds() / 3600.0,
                    "time_bucket": self._bucket_label(bucket),
                    "p_mid": float(p_mid),
                    "bid": float(bid) if not pd.isna(bid) else np.nan,
                    "ask": float(ask) if not pd.isna(ask) else np.nan,
                    "spread": float(spread) if not pd.isna(spread) else 0.0,
                    "y": label,
                    **feature_dict,
                    **sentiment_features,
                }
            )

        # Ensure all sentiment columns exist in every row for downstream consistency
        if rows and self.use_sentiment:
            for row in rows:
                for col in SENTIMENT_COLUMNS:
                    row.setdefault(col, np.nan)

        return rows

    def _prepare_trades(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        df = trades_df.copy()
        if "timestamp" not in df.columns:
            raise KeyError("Trades dataframe must include a 'timestamp' column")

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        if "price" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
        if "size" in df.columns:
            df["size"] = pd.to_numeric(df["size"], errors="coerce")

        return df

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _resolution_timestamp(self, market: Dict) -> pd.Timestamp:
        ts = market.get("resolutionTime") or market.get("endDate") or market.get(
            "resolvedTime"
        )
        if ts is None:
            raise KeyError("resolutionTime/endDate is required for snapshot labeling")
        return pd.to_datetime(ts)

    def _resolution_label(self, market: Dict) -> int:
        outcome = (
            str(
                market.get("winningOutcome")
                or market.get("resolution")
                or market.get("resolvedOutcome")
                or ""
            )
            .strip()
            .lower()
        )

        if outcome in {"yes", "y", "1", "true", "resolved_yes"}:
            return 1
        if outcome in {"no", "n", "0", "false", "resolved_no"}:
            return 0
        raise ValueError(f"Unrecognized resolution outcome: {outcome}")

    def _midpoint(self, latest_row: pd.Series, bid: float, ask: float) -> float:
        if not pd.isna(bid) and not pd.isna(ask) and bid > 0 and ask > 0:
            return float((bid + ask) / 2)
        if "price" in latest_row:
            return float(latest_row["price"])
        return 0.5

    def _spread(self, bid: float, ask: float, fallback: Optional[float]) -> float:
        if not pd.isna(bid) and not pd.isna(ask) and bid > 0 and ask > 0:
            return float(abs(ask - bid))
        if fallback is not None:
            try:
                return float(fallback)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def _window_delta(self, label: str) -> timedelta:
        if label.endswith("h"):
            hours = int(label.replace("h", ""))
            return timedelta(hours=hours)
        if label.endswith("d"):
            days = int(label.replace("d", ""))
            return timedelta(days=days)
        raise ValueError(f"Unknown sentiment window label: {label}")

    def _bucket_label(self, bucket: timedelta) -> str:
        hours = int(bucket.total_seconds() // 3600)
        days, remainder = divmod(hours, 24)
        if days > 0:
            return f"{days}d"
        return f"{remainder}h"

    def _market_features_at_snapshot(self, market: Dict, price_hint: float) -> Dict:
        market_copy = dict(market)
        market_copy["outcomePrices"] = [price_hint, 1 - price_hint]
        return self.feature_extractor.extract_market_features(market_copy)

    def _sentiment_features_from_store(
        self, market_id: str, market: Dict, snapshot_ts: datetime
    ) -> Dict[str, float]:
        # Prefer local store; optionally backfill via online providers
        features: Dict[str, float] = {}
        total_windows = 0
        windows_with_docs = 0

        if self.sentiment_store:
            for label in SENTIMENT_WINDOWS:
                delta = self._window_delta(label)
                total_windows += 1
                start_ts = int((snapshot_ts - delta).timestamp())
                end_ts = int(snapshot_ts.timestamp())
                docs = self.sentiment_store.fetch_docs(
                    market_id=market_id, start_ts=start_ts, end_ts=end_ts
                )
                scores = [
                    row["sentiment_score"]
                    for row in docs
                    if row["sentiment_score"] is not None
                    and row["published_ts"] is not None
                    and int(row["published_ts"]) <= end_ts
                ]
                stats = aggregate_scores(scores)
                features[f"sent_mean_{label}"] = stats["mean"]
                features[f"sent_std_{label}"] = stats["std"]
                features[f"doc_count_{label}"] = stats["count"]
                if stats["count"]:
                    windows_with_docs += 1

            features["sent_trend"] = features.get("sent_mean_1h", np.nan) - features.get(
                "sent_mean_24h", np.nan
            )

        missing_windows = [
            label for label in SENTIMENT_COLUMNS if np.isnan(features.get(label, np.nan))
        ]
        if (
            self.allow_online_sentiment_fetch
            and getattr(self, "sentiment_builder", None)
            and missing_windows
        ):
            try:
                online_features = canonicalize_sentiment_features(
                    self.sentiment_builder.build_features(market=market, as_of=snapshot_ts)
                )
                features.update({k: v for k, v in online_features.items() if k in SENTIMENT_COLUMNS})
            except Exception as exc:  # pragma: no cover - defensive online path
                self.logger.warning("Online sentiment backfill failed: %s", exc)

        with_docs, total = self.sentiment_coverage
        self.sentiment_coverage = (with_docs + windows_with_docs, total + total_windows)

        return canonicalize_sentiment_features(features)

    def _fetch_resolved_markets(
        self, limit: int = 50, min_volume: float = 0
    ) -> List[Dict]:
        markets = self.fetcher.get_markets(
            limit=limit, active=False, closed=True, order="volume24hr", ascending=False
        )
        resolved = []
        for market in markets:
            volume = float(market.get("volume24hr", 0) or 0)
            if volume < min_volume:
                continue
            try:
                _ = self._resolution_label(market)
                _ = self._resolution_timestamp(market)
            except Exception:
                continue
            resolved.append(market)
        return resolved


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build Polymarket snapshot dataset")
    parser.add_argument(
        "--out",
        required=True,
        help="Output Parquet path (e.g., data/features/snapshots.parquet)",
    )
    parser.add_argument("--limit", type=int, default=50, help="Number of markets")
    parser.add_argument(
        "--min-volume", type=float, default=0, help="Minimum 24h volume filter"
    )
    parser.add_argument(
        "--fidelity",
        type=int,
        default=60,
        help="Minute fidelity for price history when fetching remotely",
    )
    parser.add_argument(
        "--sentiment-db",
        type=Path,
        default=Path("data/sentiment.db"),
        help="Path to local sentiment SQLite database",
    )
    parser.add_argument(
        "--allow_online_sentiment_fetch",
        action="store_true",
        help="Allow online sentiment provider fetch if local store missing",
    )

    args = parser.parse_args(argv)

    builder = SnapshotBuilder(
        sentiment_db_path=args.sentiment_db,
        allow_online_sentiment_fetch=args.allow_online_sentiment_fetch,
    )
    dataset = builder.fetch_and_build(
        limit=args.limit, min_volume=args.min_volume, fidelity=args.fidelity
    )

    saved_path = builder.save(dataset, Path(args.out))
    print(f"Saved {len(dataset)} snapshots to {saved_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
