"""Background sentiment ingestion service."""
from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from data.sentiment_config import load_sentiment_config
from data.sentiment_features import SentimentFeatureBuilder, SentimentScorer
from data.sentiment_providers import SentimentProvider, build_providers_from_config
from polymarket_fetcher import PolymarketFetcher
from sentiment.store import DocumentStore, aggregate_scores
from sentiment.quota_store import TwitterQuotaStore

LOGGER = logging.getLogger(__name__)

DEFAULT_POLL_SEC = 300


class SentimentIngestService:
    def __init__(
        self,
        store: DocumentStore,
        providers: Iterable[SentimentProvider],
        scorer: SentimentScorer,
        fetcher: Optional[PolymarketFetcher] = None,
        poll_interval_sec: int = DEFAULT_POLL_SEC,
        backoff_sec: int = 30,
    ):
        self.store = store
        self.providers = list(providers)
        self.scorer = scorer
        self.fetcher = fetcher or PolymarketFetcher()
        self.poll_interval_sec = poll_interval_sec
        self.backoff_sec = backoff_sec
        self.feature_builder = SentimentFeatureBuilder(
            providers=self.providers, scorer=self.scorer, enabled=True
        )
        self.twitter_quota_store = TwitterQuotaStore(store.db_path)
        self._twitter_grant_day: Optional[str] = None
        self._twitter_granted_cache: Set[str] = set()

    # ------------------------------------------------------------------
    def compute_attention_score(self, market: Dict) -> float:
        volume = float(market.get("volume24hr") or market.get("volume") or 0.0)
        spread = float(market.get("spread") or market.get("bestAsk") or 0.0)
        liquidity_bonus = float(market.get("liquidity") or 0.0)
        return volume + liquidity_bonus - 0.1 * spread

    def _cluster_key(self, market: Dict) -> str:
        queries = self.feature_builder.generate_queries(market)
        if queries:
            return queries[0]
        title = (market.get("question") or market.get("title") or "").lower()
        return " ".join(title.split()[:6])

    def cluster_markets(self, markets: List[Dict]) -> Dict[str, List[Dict]]:
        clusters: Dict[str, List[Dict]] = defaultdict(list)
        for market in markets:
            clusters[self._cluster_key(market)].append(market)
        return clusters

    def _dedupe_docs(self, docs: List[Dict]) -> List[Dict]:
        seen = set()
        unique_docs = []
        for doc in docs:
            key = (doc.get("provider"), doc.get("doc_id"), doc.get("url"))
            if key in seen:
                continue
            seen.add(key)
            unique_docs.append(doc)
        return unique_docs

    def _other_docs_24h(self, market_id: Optional[str], as_of: datetime) -> int:
        if not market_id:
            return 0
        start_ts = int((as_of - timedelta(hours=24)).timestamp())
        docs = self.store.fetch_docs(market_id, start_ts=start_ts, end_ts=int(as_of.timestamp()))
        return len([d for d in docs if d["provider"] != "twitter"])

    def fetch_docs_for_market(
        self,
        market: Dict,
        as_of: datetime,
        twitter_allowlist: Optional[Set[str]] = None,
        twitter_coverage_blocks: Optional[Set[str]] = None,
        twitter_other_docs: Optional[Dict[str, int]] = None,
        twitter_day_key: Optional[str] = None,
    ) -> List[Dict]:
        all_docs: List[Dict] = []
        queries = self.feature_builder.generate_queries(market)
        if not queries:
            return []

        market_id = market.get("id") or market.get("conditionId")
        other_docs_24h = (
            (twitter_other_docs or {}).get(market_id)
            if twitter_other_docs is not None
            else None
        )
        if other_docs_24h is None:
            other_docs_24h = self._other_docs_24h(market_id, as_of)

        def _mark_twitter_granted() -> None:
            if not twitter_day_key or not market_id:
                return
            if self._twitter_grant_day != twitter_day_key:
                self._twitter_grant_day = twitter_day_key
                self._twitter_granted_cache = set(
                    self.twitter_quota_store.get_granted_markets(twitter_day_key)
                )
            if market_id in self._twitter_granted_cache:
                return
            inserted = self.twitter_quota_store.mark_market_granted(
                twitter_day_key, market_id
            )
            if inserted:
                self._twitter_granted_cache.add(market_id)

        for window in [timedelta(days=7), timedelta(days=1), timedelta(hours=6), timedelta(hours=1)]:
            start_time = as_of - window
            for provider in self.providers:
                if provider.name == "twitter" and twitter_allowlist is not None:
                    if market_id not in twitter_allowlist:
                        reason = (
                            "other_coverage"
                            if twitter_coverage_blocks and market_id in twitter_coverage_blocks
                            else "not_allowlisted"
                        )
                        LOGGER.info(
                            "twitter_fetch_skip: %s",
                            {"market_id": market_id, "reason": reason},
                        )
                        continue
                    _mark_twitter_granted()
                for query in queries:
                    try:
                        docs = provider.fetch(
                            query,
                            start_time=start_time,
                            end_time=as_of,
                            market_id=market_id,
                            other_docs_24h_count=other_docs_24h,
                        )
                        for idx, doc in enumerate(docs):
                            all_docs.append(
                                {
                                    "provider": provider.name,
                                    "doc_id": doc.get("id") or f"{int(as_of.timestamp())}_{idx}",
                                    "market_id": market.get("id") or market.get("conditionId"),
                                    "cluster_id": market.get("eventId"),
                                    "url": doc.get("url"),
                                    "title": doc.get("title"),
                                    "text": doc.get("text", ""),
                                    "published_ts": self._safe_ts(doc.get("published_at") or doc.get("publishedAt") or doc.get("seendate")),
                                    "sentiment_score": self.scorer.score(doc.get("text", "")),
                                    "raw_json": doc,
                                }
                            )
                    except Exception as exc:  # pragma: no cover - provider defensive
                        LOGGER.warning("Provider %s failed: %s", provider.name, exc)
                        continue
        return self._dedupe_docs(all_docs)

    def _safe_ts(self, value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                return int(value)
            # Try parse common formats
            return int(datetime.fromisoformat(str(value).replace("Z", "")).timestamp())
        except Exception:
            return None

    def _bucket_ts(self, as_of: datetime) -> int:
        return int(as_of.replace(minute=0, second=0, microsecond=0).timestamp())

    def _build_twitter_allowlist(
        self, markets: List[Dict], as_of: datetime
    ) -> Tuple[Set[str], Dict[str, int], Set[str]]:
        twitter_provider = next((p for p in self.providers if p.name == "twitter"), None)
        if not twitter_provider or not getattr(twitter_provider, "enabled", False):
            return set(), {}, set()

        day_key = as_of.strftime("%Y-%m-%d")
        existing = self.twitter_quota_store.get_granted_markets(day_key)
        self._twitter_grant_day = day_key
        self._twitter_granted_cache = set(existing)

        max_markets = int(
            getattr(twitter_provider, "config", {}).get(
                "max_markets_per_day", getattr(twitter_provider, "DEFAULTS", {}).get("max_markets_per_day", 3)
            )
        )
        remaining_slots = max(0, max_markets - len(existing))
        threshold = int(
            getattr(twitter_provider, "config", {}).get(
                "min_other_docs_24h", getattr(twitter_provider, "DEFAULTS", {}).get("min_other_docs_24h", 3)
            )
        )

        allowlist: Set[str] = set()
        coverage_blocks: Set[str] = set()
        other_docs: Dict[str, int] = {}
        for market in markets:
            if remaining_slots <= 0:
                break
            market_id = market.get("id") or market.get("conditionId")
            if not market_id:
                continue
            other_docs_count = self._other_docs_24h(market_id, as_of)
            other_docs[market_id] = other_docs_count
            if other_docs_count >= threshold:
                coverage_blocks.add(market_id)
                continue
            allowlist.add(market_id)
            remaining_slots -= 1

        day_remaining = max(0, max_markets - (len(existing) + len(allowlist)))
        LOGGER.info(
            "twitter_allowlist_summary: %s",
            {
                "twitter_allowlist_size": len(allowlist),
                "twitter_day_remaining": day_remaining,
                "skipped_due_to_other_coverage": len(coverage_blocks),
            },
        )
        return allowlist, other_docs, coverage_blocks

    def update_aggregates(self, market_id: str, as_of: datetime) -> None:
        bucket_ts = self._bucket_ts(as_of)
        windows = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
        }
        aggs: Dict[str, float] = {}
        for label, delta in windows.items():
            start_ts = int((as_of - delta).timestamp())
            docs = self.store.fetch_docs(market_id, start_ts=start_ts, end_ts=int(as_of.timestamp()))
            scores = [row["sentiment_score"] for row in docs if row["sentiment_score"] is not None]
            stats = aggregate_scores(scores)
            aggs[f"sent_mean_{label}"] = stats["mean"]
            aggs[f"sent_std_{label}"] = stats["std"]
            aggs[f"doc_count_{label}"] = stats["count"]
        aggs["sent_trend"] = aggs.get("sent_mean_1h", np.nan) - aggs.get("sent_mean_24h", np.nan)
        self.store.upsert_aggregate(market_id=market_id, bucket_ts=bucket_ts, agg=aggs)

    def run_once(self, markets: Optional[List[Dict]] = None, as_of: Optional[datetime] = None) -> None:
        as_of = as_of or datetime.utcnow()
        markets = markets or self.fetcher.get_markets(limit=200, active=True, closed=False)
        clusters = self.cluster_markets(markets)
        attention_by_cluster: List[Tuple[str, float, List[Dict]]] = []
        for key, mkts in clusters.items():
            score = max(self.compute_attention_score(m) for m in mkts)
            attention_by_cluster.append((key, score, mkts))

        attention_by_cluster.sort(key=lambda t: t[1], reverse=True)
        ordered_markets: List[Dict] = []
        for _, _, mkts in attention_by_cluster:
            ordered_markets.extend(mkts)

        twitter_allowlist, twitter_other_docs, twitter_coverage_blocks = self._build_twitter_allowlist(
            ordered_markets, as_of
        )
        day_key = as_of.strftime("%Y-%m-%d") if twitter_allowlist or twitter_coverage_blocks else None
        for _, _, mkts in attention_by_cluster:
            for market in mkts:
                docs = self.fetch_docs_for_market(
                    market,
                    as_of=as_of,
                    twitter_allowlist=twitter_allowlist,
                    twitter_coverage_blocks=twitter_coverage_blocks,
                    twitter_other_docs=twitter_other_docs,
                    twitter_day_key=day_key,
                )
                inserted = self.store.upsert_documents(docs) if docs else 0
                if inserted:
                    LOGGER.info("Stored %s new docs for %s", inserted, market.get("id"))
                market_id = market.get("id") or market.get("conditionId")
                if market_id:
                    self.update_aggregates(market_id=market_id, as_of=as_of)

    def run_forever(self) -> None:
        while True:
            try:
                self.run_once()
                time.sleep(self.poll_interval_sec)
            except KeyboardInterrupt:  # pragma: no cover - manual stop
                LOGGER.info("Sentiment ingest stopped by user")
                break
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Ingest iteration failed: %s", exc)
                time.sleep(self.backoff_sec)


def build_service(db_path: Path, sentiment_config: Optional[Path] = None) -> SentimentIngestService:
    cfg = load_sentiment_config(sentiment_config)
    scorer = SentimentScorer(model_name=cfg.get("sentiment", {}).get("model"))
    providers = build_providers_from_config(cfg)
    store = DocumentStore(db_path)
    return SentimentIngestService(store=store, providers=providers, scorer=scorer)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Background sentiment ingestion service")
    parser.add_argument("--db", type=Path, default=Path("data/sentiment.db"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--once", action="store_true", help="Run a single iteration and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    service = build_service(args.db, args.config)
    if args.once:
        service.run_once()
    else:
        service.run_forever()


if __name__ == "__main__":
    main()
