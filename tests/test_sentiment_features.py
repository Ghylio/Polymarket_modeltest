import unittest
from datetime import datetime, timedelta

import unittest
from datetime import datetime, timedelta

from data.sentiment_features import SentimentFeatureBuilder, SentimentScorer
from data.sentiment_providers import build_providers_from_config, SentimentProvider


class DummyProvider(SentimentProvider):
    name = "dummy"

    def __init__(self, docs):
        super().__init__(enabled=True)
        self.docs = docs

    def fetch(self, query, start_time, end_time):
        return [doc for doc in self.docs if start_time <= doc["ts"] <= end_time]


class DummyScorer(SentimentScorer):
    def __init__(self):
        super().__init__(model_name=None)

    def score(self, text: str) -> float:  # type: ignore[override]
        return 1.0 if "good" in text else 0.2


class SentimentFeatureTests(unittest.TestCase):
    def setUp(self):
        now = datetime(2024, 1, 2, 12, 0, 0)
        self.now = now
        self.docs = [
            {"text": "very good news", "ts": now - timedelta(hours=1)},
            {"text": "bad outcome", "ts": now - timedelta(hours=2)},
            {"text": "good trend continues", "ts": now - timedelta(hours=20)},
            {"text": "neutral", "ts": now - timedelta(days=3)},
        ]

    def test_aggregation_and_trend(self):
        provider = DummyProvider(self.docs)
        builder = SentimentFeatureBuilder(
            providers=[provider], scorer=DummyScorer(), enabled=True
        )
        market = {"title": "Sample Market", "description": ""}
        feats = builder.build_features(market, as_of=self.now)
        self.assertGreater(feats["sent_mean_1h"], 0)
        self.assertAlmostEqual(feats["doc_count_24h"], 3)
        self.assertIn("sent_trend", feats)
        self.assertNotEqual(feats["sent_trend"], 0.0)

    def test_provider_disables_when_keys_missing(self):
        cfg = {
            "providers": {
                "gdelt": {"enabled": False},
                "newsapi": {"enabled": True, "api_key": ""},
                "twitter": {"enabled": True, "bearer_token": ""},
            },
            "sentiment": {"enabled": True, "model": None},
        }
        providers = build_providers_from_config(cfg)
        self.assertEqual(len(providers), 0)

    def test_enabled_without_providers_returns_nans(self):
        builder = SentimentFeatureBuilder(providers=[], scorer=DummyScorer(), enabled=True)
        market = {"title": "No providers", "description": ""}
        feats = builder.build_features(market, as_of=self.now)
        self.assertIn("sent_mean_1h", feats)
        self.assertTrue(all(value != value for value in feats.values()))


if __name__ == "__main__":
    unittest.main()
