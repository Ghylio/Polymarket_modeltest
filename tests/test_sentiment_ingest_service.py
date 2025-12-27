import math
from datetime import datetime, timedelta
from typing import List

from sentiment.ingest_service import SentimentIngestService
from sentiment.store import DocumentStore


class MockProvider:
    name = "mock"

    def __init__(self):
        self.calls = 0

    def fetch(self, query, start_time, end_time):  # noqa: D401 - test mock
        self.calls += 1
        return [
            {
                "id": "doc1",
                "text": "great outlook",
                "published_at": datetime.utcnow().isoformat(),
            }
        ]


class MockFetcher:
    def __init__(self, markets: List[dict] | None = None):
        self._markets = markets or [
            {
                "id": "m1",
                "question": "Will Team A win?",
                "volume24hr": 1000,
                "spread": 0.02,
            }
        ]

    def get_markets(self, **kwargs):  # noqa: D401 - test helper
        return self._markets


class ZeroScorer:
    def __init__(self):
        self.model_name = "zero"

    def score(self, text: str) -> float:  # noqa: D401
        return 0.5


class SkipProvider:
    name = "skip"

    def fetch(self, query, start_time, end_time, **kwargs):  # noqa: D401 - test mock
        return []


class MockTwitterProvider:
    name = "twitter"

    def __init__(self, config=None, enabled=True):
        self.enabled = enabled
        self.config = {
            "max_markets_per_day": 2,
            "min_other_docs_24h": 1,
        }
        if config:
            self.config.update(config)
        self.calls = []

    def fetch(self, query, start_time, end_time, **kwargs):  # noqa: D401 - test mock
        self.calls.append(kwargs.get("market_id"))
        return [
            {
                "id": f"tw_{kwargs.get('market_id', 'unknown')}",
                "text": query,
                "published_at": datetime.utcnow().isoformat(),
            }
        ]


def test_service_single_iteration(tmp_path):
    store = DocumentStore(tmp_path / "sentiment.db")
    provider = MockProvider()
    service = SentimentIngestService(
        store=store,
        providers=[provider],
        scorer=ZeroScorer(),
        fetcher=MockFetcher(),
        poll_interval_sec=1,
    )
    as_of = datetime.utcnow()
    service.run_once(as_of=as_of)
    docs = store.fetch_docs("m1", start_ts=int(as_of.timestamp()) - 3600, end_ts=int(as_of.timestamp()) + 10)
    assert len(docs) == 1
    agg = store.fetch_aggregate("m1", bucket_ts=service._bucket_ts(as_of))
    assert agg is not None
    assert agg["doc_count_1h"] == 1
    assert provider.calls > 0


def test_service_writes_nan_aggregates_when_no_docs(tmp_path):
    store = DocumentStore(tmp_path / "sentiment.db")
    provider = SkipProvider()
    service = SentimentIngestService(
        store=store,
        providers=[provider],
        scorer=ZeroScorer(),
        fetcher=MockFetcher(),
        poll_interval_sec=1,
    )
    as_of = datetime.utcnow()
    service.run_once(as_of=as_of)
    agg = store.fetch_aggregate("m1", bucket_ts=service._bucket_ts(as_of))
    assert agg is not None
    assert agg["doc_count_1h"] == 0
    assert math.isnan(float(agg["sent_mean_1h"]))


def test_twitter_allowlist_top_k(tmp_path):
    store = DocumentStore(tmp_path / "sentiment.db")
    twitter_provider = MockTwitterProvider(config={"max_markets_per_day": 2, "min_other_docs_24h": 10})
    markets = [
        {"id": "m1", "question": "Q1", "volume24hr": 500, "spread": 0.1},
        {"id": "m2", "question": "Q2", "volume24hr": 400, "spread": 0.1},
        {"id": "m3", "question": "Q3", "volume24hr": 300, "spread": 0.1},
    ]
    service = SentimentIngestService(
        store=store,
        providers=[twitter_provider],
        scorer=ZeroScorer(),
        fetcher=MockFetcher(markets=markets),
        poll_interval_sec=1,
    )
    service.run_once(as_of=datetime.utcnow())
    assert set(twitter_provider.calls) == {"m1", "m2"}


def test_twitter_allowlist_skips_when_other_coverage(tmp_path):
    store = DocumentStore(tmp_path / "sentiment.db")
    as_of = datetime.utcnow()
    store.upsert_documents(
        [
            {
                "provider": "gdelt",
                "doc_id": "d1",
                "market_id": "m1",
                "cluster_id": "c1",
                "published_ts": int((as_of - timedelta(hours=1)).timestamp()),
                "sentiment_score": 0.1,
            }
        ]
    )
    twitter_provider = MockTwitterProvider(config={"max_markets_per_day": 2, "min_other_docs_24h": 1})
    markets = [
        {"id": "m1", "question": "Q1", "volume24hr": 500, "spread": 0.1},
        {"id": "m2", "question": "Q2", "volume24hr": 400, "spread": 0.1},
    ]
    service = SentimentIngestService(
        store=store,
        providers=[twitter_provider],
        scorer=ZeroScorer(),
        fetcher=MockFetcher(markets=markets),
        poll_interval_sec=1,
    )
    service.run_once(as_of=as_of)
    assert twitter_provider.calls == ["m2"]


def test_twitter_allowlist_respects_persisted_grants(tmp_path):
    store_path = tmp_path / "sentiment.db"
    store = DocumentStore(store_path)
    twitter_provider = MockTwitterProvider(config={"max_markets_per_day": 1, "min_other_docs_24h": 10})
    markets = [
        {"id": "m1", "question": "Q1", "volume24hr": 500, "spread": 0.1},
        {"id": "m2", "question": "Q2", "volume24hr": 400, "spread": 0.1},
    ]
    as_of = datetime.utcnow()
    service = SentimentIngestService(
        store=store,
        providers=[twitter_provider],
        scorer=ZeroScorer(),
        fetcher=MockFetcher(markets=markets),
        poll_interval_sec=1,
    )
    service.run_once(as_of=as_of)
    assert twitter_provider.calls == ["m1"]

    # second run same day should see grant persisted and skip additional markets
    twitter_provider.calls.clear()
    service2 = SentimentIngestService(
        store=DocumentStore(store_path),
        providers=[twitter_provider],
        scorer=ZeroScorer(),
        fetcher=MockFetcher(markets=markets),
        poll_interval_sec=1,
    )
    service2.run_once(as_of=as_of)
    assert twitter_provider.calls == []
