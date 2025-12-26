import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import unittest

from research.ingest_service import ResearchIngestService
from research.prompt import parse_llm_output
from research.store import ResearchFeatureStore
from sentiment.store import DocumentStore


class DummyLLMClient:
    def __init__(self):
        self.calls = 0
        self._cache = {}

    def has_cache(self, market_id, bucket):
        return (market_id, bucket) in self._cache

    def call_llm(self, market_id, bucket, prompt, schema):  # noqa: ARG002
        self.calls += 1
        self._cache[(market_id, bucket)] = True
        return {"llm_p_yes": 0.6, "llm_confidence": 0.7, "resolution_source_type": "news"}, False


class DummyFetcher:
    def __init__(self, markets):
        self.markets = markets

    def get_markets(self, **_):
        return self.markets


class ResearchIngestTests(unittest.TestCase):
    def test_prompt_parse_to_schema(self):
        raw = """{""" + ",".join(
            [
                '"llm_p_yes":0.55',
                '"llm_confidence":0.8',
                '"evidence_count_24h":3',
                '"stance_score_24h":null'
            ]
        ) + "}"""
        features = parse_llm_output(raw)
        self.assertAlmostEqual(features.llm_p_yes, 0.55)
        self.assertIsNone(features.stance_score_24h)

    def test_ingest_one_iteration_with_docs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sentiment.db"
            doc_store = DocumentStore(db_path)
            research_store = ResearchFeatureStore(db_path)

            now = int(time.time())
            doc_store.upsert_documents(
                [
                    {
                        "provider": "news",
                        "doc_id": "d1",
                        "market_id": "m1",
                        "title": "Update",
                        "text": "Market moves",
                        "published_ts": now - 3600,
                    }
                ]
            )

            markets = [
                {
                    "id": "m1",
                    "question": "Will it rain?",
                    "volume24hr": 100,
                    "spread": 0.1,
                }
            ]
            llm = DummyLLMClient()
            service = ResearchIngestService(
                research_store,
                doc_store,
                llm,
                fetcher=DummyFetcher(markets),
                config={
                    "window_hours_short": 24,
                    "window_hours_long": 168,
                    "top_k": 5,
                    "use_embeddings": False,
                    "exploration_rate": 0.0,
                },
                max_markets=5,
            )

            service.run_once(as_of_ts=now)
            features, found = research_store.fetch_latest_features("m1", now, return_found=True)
            self.assertTrue(found)
            self.assertIn("llm_p_yes", features)

    def test_llm_cache_prevents_double_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sentiment.db"
            doc_store = DocumentStore(db_path)
            research_store = ResearchFeatureStore(db_path)

            now = int(time.time())
            doc_store.upsert_documents(
                [
                    {
                        "provider": "news",
                        "doc_id": "d1",
                        "market_id": "m1",
                        "title": "Update",
                        "text": "Market moves",
                        "published_ts": now - 3600,
                    }
                ]
            )

            markets = [{"id": "m1", "question": "Will it rain?"}]
            llm = DummyLLMClient()
            service = ResearchIngestService(
                research_store,
                doc_store,
                llm,
                fetcher=DummyFetcher(markets),
                config={
                    "window_hours_short": 24,
                    "window_hours_long": 168,
                    "top_k": 5,
                    "use_embeddings": False,
                    "exploration_rate": 0.0,
                },
                max_markets=5,
            )

            service.run_once(as_of_ts=now)
            service.run_once(as_of_ts=now)
            self.assertEqual(llm.calls, 1)

    def test_rules_only_path_when_no_docs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sentiment.db"
            doc_store = DocumentStore(db_path)
            research_store = ResearchFeatureStore(db_path)

            now = int(time.time())
            markets = [
                {
                    "id": "m2",
                    "question": "Will official CPI be above 3%?",
                    "rules": "Based on official published CPI release",
                }
            ]

            llm = DummyLLMClient()
            service = ResearchIngestService(
                research_store,
                doc_store,
                llm,
                fetcher=DummyFetcher(markets),
                config={
                    "window_hours_short": 24,
                    "window_hours_long": 168,
                    "top_k": 5,
                    "use_embeddings": False,
                    "exploration_rate": 0.0,
                    "min_docs_for_llm": 1,
                },
                max_markets=5,
            )

            service.run_once(as_of_ts=now)
            self.assertEqual(llm.calls, 0)
            features, found = research_store.fetch_latest_features("m2", now, return_found=True)
            self.assertTrue(found)
            self.assertEqual(features.get("resolution_source_type"), "official_statement")


if __name__ == "__main__":
    unittest.main()
