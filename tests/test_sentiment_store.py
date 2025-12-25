import os
import tempfile
from datetime import datetime

from sentiment.store import DocumentStore


def test_document_dedupe_and_fetch():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "sentiment.db")
        store = DocumentStore(db_path)
        now_ts = int(datetime.utcnow().timestamp())
        docs = [
            {
                "provider": "gdelt",
                "doc_id": "1",
                "market_id": "m1",
                "url": "http://example.com/a",
                "text": "good news",
                "published_ts": now_ts,
                "sentiment_score": 0.9,
            },
            {
                "provider": "gdelt",
                "doc_id": "1",
                "market_id": "m1",
                "url": "http://example.com/a",
                "text": "good news duplicate",
                "published_ts": now_ts,
                "sentiment_score": 0.2,
            },
        ]
        inserted = store.upsert_documents(docs)
        assert inserted == 1

        rows = store.fetch_docs("m1", start_ts=now_ts - 10, end_ts=now_ts + 10)
        assert len(rows) == 1
        assert rows[0]["sentiment_score"] == 0.9


def test_aggregate_upsert_and_fetch():
    with tempfile.TemporaryDirectory() as tmp:
        store = DocumentStore(os.path.join(tmp, "sentiment.db"))
        agg = {
            "sent_mean_1h": 0.5,
            "sent_std_1h": 0.1,
            "doc_count_1h": 2,
            "sent_mean_6h": 0.6,
            "sent_std_6h": 0.2,
            "doc_count_6h": 3,
            "sent_mean_24h": 0.55,
            "sent_std_24h": 0.15,
            "doc_count_24h": 4,
            "sent_mean_7d": 0.52,
            "sent_std_7d": 0.12,
            "doc_count_7d": 5,
            "sent_trend": 0.1,
        }
        store.upsert_aggregate("m1", bucket_ts=1000, agg=agg)
        row = store.fetch_aggregate("m1", bucket_ts=1000)
        assert row is not None
        assert row["sent_mean_6h"] == 0.6
        # Update with new values
        agg["sent_mean_6h"] = 0.8
        store.upsert_aggregate("m1", bucket_ts=1000, agg=agg)
        row = store.fetch_aggregate("m1", bucket_ts=1000)
        assert row["sent_mean_6h"] == 0.8
