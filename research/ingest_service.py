"""Background research ingestion service (RAG + LLM)."""
from __future__ import annotations

import argparse
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import yaml

from metrics.logger import MetricsLogger
from polymarket_fetcher import PolymarketFetcher
from research.llm import LLMClient
from research.prompt import build_prompt, parse_llm_output
from research.rules_only import extract_rules_features
from research.retrieval import get_market_docs
from research.schema import ResearchFeatures
from research.store import ResearchFeatureStore
from sentiment.store import DocumentStore

LOGGER = logging.getLogger(__name__)

DEFAULT_INTERVAL_SEC = 300
DEFAULT_CONFIG_PATH = Path("config/research_config.yaml")
DEFAULT_MIN_DOCS_FOR_LLM = 1


def load_research_config(path: Path | None = None) -> Dict:
    cfg_path = path or DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return {
            "model": "gpt-4o-mini",
            "top_k": 10,
            "window_hours_short": 24,
            "window_hours_long": 168,
            "use_embeddings": False,
            "exploration_rate": 0.1,
            "rate_limit_per_min": 30,
            "min_docs_for_llm": DEFAULT_MIN_DOCS_FOR_LLM,
        }
    with cfg_path.open("r") as f:
        loaded = yaml.safe_load(f) or {}
    return {
        "model": loaded.get("model", "gpt-4o-mini"),
        "top_k": int(loaded.get("top_k", 10)),
        "window_hours_short": int(loaded.get("window_hours_short", 24)),
        "window_hours_long": int(loaded.get("window_hours_long", 168)),
        "use_embeddings": bool(loaded.get("use_embeddings", False)),
        "exploration_rate": float(loaded.get("exploration_rate", 0.1)),
        "rate_limit_per_min": int(loaded.get("rate_limit_per_min", 30)),
        "min_docs_for_llm": int(loaded.get("min_docs_for_llm", DEFAULT_MIN_DOCS_FOR_LLM)),
    }


class ResearchIngestService:
    def __init__(
        self,
        research_store: ResearchFeatureStore,
        document_store: DocumentStore,
        llm_client: LLMClient,
        fetcher: Optional[PolymarketFetcher] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        config: Optional[Dict] = None,
        retrieval_fn: Callable[..., List[Dict]] = get_market_docs,
        interval_sec: int = DEFAULT_INTERVAL_SEC,
        max_markets: int = 50,
    ):
        self.research_store = research_store
        self.document_store = document_store
        self.llm_client = llm_client
        self.fetcher = fetcher or PolymarketFetcher()
        self.metrics = metrics_logger
        self.config = config or load_research_config(None)
        self.retrieval_fn = retrieval_fn
        self.interval_sec = interval_sec
        self.max_markets = max_markets

    # ------------------------------------------------------------------
    def compute_attention_score(self, market: Dict) -> float:
        volume = float(market.get("volume24hr") or market.get("volume") or 0.0)
        spread = float(market.get("spread") or market.get("bestAsk") or 0.0)
        liquidity = float(market.get("liquidity") or 0.0)
        near_resolve_bonus = 1.0 if market.get("closeTime") else 0.0
        return volume + liquidity + near_resolve_bonus - 0.1 * spread

    def select_markets(self, markets: List[Dict]) -> List[Dict]:
        markets = markets or []
        scored = [(self.compute_attention_score(m), m) for m in markets]
        scored.sort(key=lambda t: t[0], reverse=True)
        top_cut = int(max(1, self.max_markets * (1 - self.config.get("exploration_rate", 0.1))))
        top_markets = [m for _, m in scored[:top_cut]]

        remaining = [m for _, m in scored[top_cut:]]
        random.shuffle(remaining)
        exploration_count = min(len(remaining), max(1, int(self.max_markets * self.config.get("exploration_rate", 0.1))))
        exploratory = remaining[:exploration_count]
        return (top_markets + exploratory)[: self.max_markets]

    def _hour_bucket(self, ts: int) -> int:
        return int(datetime.utcfromtimestamp(ts).replace(minute=0, second=0, microsecond=0).timestamp())

    # ------------------------------------------------------------------
    def run_once(self, as_of_ts: Optional[int] = None) -> None:
        now_ts = as_of_ts or int(time.time())
        markets = self.fetcher.get_markets(limit=self.max_markets, active=True, closed=False)
        selected = self.select_markets(markets)

        for market in selected:
            market_id = market.get("id") or market.get("conditionId")
            if not market_id:
                continue
            bucket = self._hour_bucket(now_ts)
            try:
                if getattr(self.llm_client, "has_cache", lambda *_: False)(market_id, bucket):
                    self._log_metric("research_ingest_skip", market_id, bucket, reason="cached")
                    continue

                docs_short = self.retrieval_fn(
                    self.document_store,
                    market,
                    now_ts,
                    self.config.get("window_hours_short", 24),
                    self.config.get("top_k", 10),
                    self.config.get("use_embeddings", False),
                )
                docs_long = self.retrieval_fn(
                    self.document_store,
                    market,
                    now_ts,
                    self.config.get("window_hours_long", 168),
                    self.config.get("top_k", 10),
                    self.config.get("use_embeddings", False),
                )

                merged_docs = docs_short + [d for d in docs_long if d not in docs_short]

                min_docs = self.config.get("min_docs_for_llm", DEFAULT_MIN_DOCS_FOR_LLM)
                if len(merged_docs) < min_docs:
                    base = extract_rules_features(market)
                    features = self._enrich_counts(ResearchFeatures(**base), merged_docs, now_ts)
                    self.research_store.upsert_research_features(
                        market_id=market_id,
                        as_of_ts=now_ts,
                        features=features,
                        raw_json=base,
                    )
                    self._log_metric(
                        "research_rules_only",
                        market_id,
                        bucket,
                        ambiguity_score=base.get("ambiguity_score"),
                        resolution_source_type=base.get("resolution_source_type"),
                    )
                    continue

                prompt = build_prompt(market, merged_docs, now_ts)
                llm_result, from_cache = self.llm_client.call_llm(market_id, bucket, prompt)
                if from_cache:
                    self._log_metric("research_ingest_skip", market_id, bucket, reason="cached")
                    continue

                features = parse_llm_output(llm_result)
                features = self._enrich_counts(features, merged_docs, now_ts)
                self.research_store.upsert_research_features(
                    market_id=market_id,
                    as_of_ts=now_ts,
                    features=features,
                    raw_json=llm_result,
                )
                self._log_metric(
                    "research_ingest_success",
                    market_id,
                    bucket,
                    doc_count=len(merged_docs),
                    latency=0,
                )
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Research ingest failed for %s: %s", market_id, exc)
                self._log_metric("research_ingest_error", market_id, bucket, error=str(exc))

    def _enrich_counts(self, features: ResearchFeatures, docs: Iterable[Dict], now_ts: int) -> ResearchFeatures:
        evidence_24h = sum(1 for d in docs if d.get("published_ts") and d.get("published_ts") >= now_ts - 24 * 3600)
        evidence_7d = sum(1 for d in docs if d.get("published_ts") and d.get("published_ts") >= now_ts - 7 * 24 * 3600)
        sources = {d.get("source") or d.get("provider") for d in docs if d.get("source") or d.get("provider")}
        enriched = features.dict()
        if enriched.get("evidence_count_24h") is None:
            enriched["evidence_count_24h"] = evidence_24h
        if enriched.get("evidence_count_7d") is None:
            enriched["evidence_count_7d"] = evidence_7d
        if enriched.get("source_diversity_7d") is None:
            enriched["source_diversity_7d"] = len(sources)
        return ResearchFeatures(**enriched)

    def _log_metric(self, event_type: str, market_id: str, hour_bucket: int, **kwargs) -> None:
        if self.metrics:
            self.metrics.log_event(event_type, {"market_id": market_id, "hour_bucket": hour_bucket, **kwargs})

    # ------------------------------------------------------------------
    def run_forever(self) -> None:  # pragma: no cover - runtime loop
        while True:
            self.run_once()
            time.sleep(self.interval_sec)


def main():  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Run research ingestion service")
    parser.add_argument("--db", type=Path, default=Path("data/sentiment.db"))
    parser.add_argument("--interval_sec", type=int, default=DEFAULT_INTERVAL_SEC)
    parser.add_argument("--max_markets", type=int, default=50)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    cfg = load_research_config(args.config)
    research_store = ResearchFeatureStore(args.db)
    document_store = DocumentStore(args.db)
    llm_client = LLMClient(cfg.get("model"), rate_limit_per_min=cfg.get("rate_limit_per_min", 30))
    metrics_dir = Path("results") / "research_ingest"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with MetricsLogger(metrics_dir / "run") as logger:
        service = ResearchIngestService(
            research_store,
            document_store,
            llm_client,
            metrics_logger=logger,
            config=cfg,
            interval_sec=args.interval_sec,
            max_markets=args.max_markets,
        )
        service.run_forever()


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
