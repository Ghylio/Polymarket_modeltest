from datetime import datetime

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
    def get_markets(self, **kwargs):
        return [
            {
                "id": "m1",
                "question": "Will Team A win?",
                "volume24hr": 1000,
                "spread": 0.02,
            }
        ]


class ZeroScorer:
    def __init__(self):
        self.model_name = "zero"

    def score(self, text: str) -> float:  # noqa: D401
        return 0.5


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
