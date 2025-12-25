"""Local sentiment storage for documents and aggregates."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DOC_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS docs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    doc_id TEXT,
    market_id TEXT,
    cluster_id TEXT,
    url TEXT,
    url_hash TEXT,
    title TEXT,
    text TEXT,
    published_ts INTEGER,
    fetched_ts INTEGER,
    lang TEXT,
    sentiment_score REAL,
    raw_json TEXT,
    UNIQUE(provider, doc_id),
    UNIQUE(url_hash)
);
"""

AGG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS aggregates (
    market_id TEXT NOT NULL,
    bucket_ts INTEGER NOT NULL,
    sent_mean_1h REAL,
    sent_std_1h REAL,
    doc_count_1h REAL,
    sent_mean_6h REAL,
    sent_std_6h REAL,
    doc_count_6h REAL,
    sent_mean_24h REAL,
    sent_std_24h REAL,
    doc_count_24h REAL,
    sent_mean_7d REAL,
    sent_std_7d REAL,
    doc_count_7d REAL,
    sent_trend REAL,
    updated_ts INTEGER,
    PRIMARY KEY (market_id, bucket_ts)
);
"""


class DocumentStore:
    """SQLite-backed sentiment document and aggregate store."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        if self.db_path.parent and not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute(DOC_TABLE_SQL)
        cur.execute(AGG_TABLE_SQL)
        self.conn.commit()

    # ------------------------------------------------------------------
    def _hash_url(self, url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def upsert_documents(self, docs: Iterable[Dict]) -> int:
        """Insert documents with deduplication on provider+doc_id or url hash."""

        to_insert: List[Tuple] = []
        now_ts = int(time.time())
        for doc in docs:
            url_hash = self._hash_url(doc.get("url"))
            to_insert.append(
                (
                    doc.get("provider"),
                    doc.get("doc_id"),
                    doc.get("market_id"),
                    doc.get("cluster_id"),
                    doc.get("url"),
                    url_hash,
                    doc.get("title"),
                    doc.get("text"),
                    int(doc.get("published_ts")) if doc.get("published_ts") is not None else None,
                    int(doc.get("fetched_ts", now_ts)),
                    doc.get("lang"),
                    doc.get("sentiment_score"),
                    json.dumps(doc.get("raw_json")) if doc.get("raw_json") is not None else None,
                )
            )

        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT OR IGNORE INTO docs (
                provider, doc_id, market_id, cluster_id, url, url_hash, title, text,
                published_ts, fetched_ts, lang, sentiment_score, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            to_insert,
        )
        self.conn.commit()
        return cur.rowcount

    def fetch_docs(
        self, market_id: Optional[str], start_ts: int, end_ts: int
    ) -> List[sqlite3.Row]:
        cur = self.conn.cursor()
        if market_id:
            cur.execute(
                """
                SELECT * FROM docs
                WHERE market_id = ? AND published_ts BETWEEN ? AND ?
            """,
                (market_id, start_ts, end_ts),
            )
        else:
            cur.execute(
                """
                SELECT * FROM docs
                WHERE published_ts BETWEEN ? AND ?
            """,
                (start_ts, end_ts),
            )
        return cur.fetchall()

    # ------------------------------------------------------------------
    def upsert_aggregate(self, market_id: str, bucket_ts: int, agg: Dict[str, float]) -> None:
        fields = [
            "sent_mean_1h",
            "sent_std_1h",
            "doc_count_1h",
            "sent_mean_6h",
            "sent_std_6h",
            "doc_count_6h",
            "sent_mean_24h",
            "sent_std_24h",
            "doc_count_24h",
            "sent_mean_7d",
            "sent_std_7d",
            "doc_count_7d",
            "sent_trend",
        ]
        values = [agg.get(f) for f in fields]
        values.extend([market_id, bucket_ts, int(time.time())])
        placeholders = ", ".join("?" for _ in fields)
        cur = self.conn.cursor()
        cur.execute(
            f"""
            INSERT INTO aggregates ({', '.join(['sent_mean_1h','sent_std_1h','doc_count_1h','sent_mean_6h','sent_std_6h','doc_count_6h','sent_mean_24h','sent_std_24h','doc_count_24h','sent_mean_7d','sent_std_7d','doc_count_7d','sent_trend','market_id','bucket_ts','updated_ts'])})
            VALUES ({placeholders}, ?, ?, ?)
            ON CONFLICT(market_id, bucket_ts) DO UPDATE SET
                sent_mean_1h=excluded.sent_mean_1h,
                sent_std_1h=excluded.sent_std_1h,
                doc_count_1h=excluded.doc_count_1h,
                sent_mean_6h=excluded.sent_mean_6h,
                sent_std_6h=excluded.sent_std_6h,
                doc_count_6h=excluded.doc_count_6h,
                sent_mean_24h=excluded.sent_mean_24h,
                sent_std_24h=excluded.sent_std_24h,
                doc_count_24h=excluded.doc_count_24h,
                sent_mean_7d=excluded.sent_mean_7d,
                sent_std_7d=excluded.sent_std_7d,
                doc_count_7d=excluded.doc_count_7d,
                sent_trend=excluded.sent_trend,
                updated_ts=excluded.updated_ts
            """,
            values,
        )
        self.conn.commit()

    def fetch_aggregate(self, market_id: str, bucket_ts: int) -> Optional[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM aggregates WHERE market_id = ? AND bucket_ts = ?",
            (market_id, bucket_ts),
        )
        return cur.fetchone()

    def close(self) -> None:
        self.conn.close()


def aggregate_scores(scores: Sequence[float]) -> Dict[str, float]:
    arr = np.array(scores, dtype=float)
    if arr.size == 0:
        return {"mean": np.nan, "std": np.nan, "count": 0.0}
    return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)), "count": float(arr.size)}
