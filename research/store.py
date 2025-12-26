from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from research.schema import (
    RESEARCH_COLUMNS,
    ResearchFeatures,
    canonicalize_research_features,
    default_research_features,
)

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS research_features (
    market_id TEXT NOT NULL,
    as_of_ts INTEGER NOT NULL,
    hour_bucket INTEGER NOT NULL,
    llm_p_yes REAL,
    llm_confidence REAL,
    evidence_count_24h INTEGER,
    evidence_count_7d INTEGER,
    source_diversity_7d INTEGER,
    stance_score_24h REAL,
    stance_score_7d REAL,
    rules_ambiguous INTEGER,
    resolution_source_type TEXT,
    ambiguity_score REAL,
    updated_ts INTEGER,
    raw_json TEXT,
    UNIQUE(market_id, hour_bucket)
);
"""


class ResearchFeatureStore:
    """SQLite store for leakage-safe research features."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        if self.db_path.parent and not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute(TABLE_SQL)
        self.conn.commit()

    # ------------------------------------------------------------------
    def upsert_research_features(
        self,
        market_id: str,
        as_of_ts: int,
        features: ResearchFeatures,
        raw_json: Optional[Any] = None,
    ) -> None:
        """Insert or update the research feature row for the hour bucket."""

        bucket = self._hour_bucket(as_of_ts)
        canonical = canonicalize_research_features(features.dict())
        payload = [canonical.get(col) for col in RESEARCH_COLUMNS]
        payload.extend(
            [
                market_id,
                int(as_of_ts),
                int(bucket),
                int(time.time()),
                json.dumps(raw_json) if raw_json is not None else None,
            ]
        )

        placeholders = ", ".join("?" for _ in RESEARCH_COLUMNS)
        cur = self.conn.cursor()
        cur.execute(
            f"""
            INSERT INTO research_features (
                llm_p_yes, llm_confidence, evidence_count_24h, evidence_count_7d,
                source_diversity_7d, stance_score_24h, stance_score_7d, rules_ambiguous,
                resolution_source_type, ambiguity_score, market_id, as_of_ts, hour_bucket, updated_ts, raw_json
            ) VALUES ({placeholders}, ?, ?, ?, ?, ?)
            ON CONFLICT(market_id, hour_bucket) DO UPDATE SET
                llm_p_yes=excluded.llm_p_yes,
                llm_confidence=excluded.llm_confidence,
                evidence_count_24h=excluded.evidence_count_24h,
                evidence_count_7d=excluded.evidence_count_7d,
                source_diversity_7d=excluded.source_diversity_7d,
                stance_score_24h=excluded.stance_score_24h,
                stance_score_7d=excluded.stance_score_7d,
                rules_ambiguous=excluded.rules_ambiguous,
                resolution_source_type=excluded.resolution_source_type,
                ambiguity_score=excluded.ambiguity_score,
                as_of_ts=excluded.as_of_ts,
                updated_ts=excluded.updated_ts,
                raw_json=excluded.raw_json
            """,
            payload,
        )
        self.conn.commit()

    def fetch_latest_features(
        self, market_id: str, cutoff_ts: int, return_found: bool = False
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], bool]]:
        """
        Return the most recent feature row with as_of_ts <= cutoff_ts.

        If none exist, return defaults with NaN/0 fills. When return_found is
        True, a boolean flag indicating whether a row was found is returned as
        the second tuple element.
        """

        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM research_features
            WHERE market_id = ? AND as_of_ts <= ?
            ORDER BY as_of_ts DESC
            LIMIT 1
            """,
            (market_id, int(cutoff_ts)),
        )
        row = cur.fetchone()
        if row is None:
            default = default_research_features()
            return (default, False) if return_found else default

        features = canonicalize_research_features(dict(row))
        return (features, True) if return_found else features

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def _hour_bucket(as_of_ts: int) -> int:
        dt = datetime.utcfromtimestamp(int(as_of_ts)).replace(minute=0, second=0, microsecond=0)
        return int(dt.timestamp())
