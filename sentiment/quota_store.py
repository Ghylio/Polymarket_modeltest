"""SQLite-backed quota and cache store for Twitter/X usage.

This module is intentionally minimal to avoid adding heavier dependencies.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional, Tuple


QUOTA_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS twitter_quota_state (
    month_key TEXT PRIMARY KEY,
    reads_used INTEGER,
    budget INTEGER
);
"""

CACHE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS twitter_query_cache (
    cache_key TEXT PRIMARY KEY,
    created_ts INTEGER,
    response_json TEXT
);
"""

COOLDOWN_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS twitter_market_cooldown (
    market_id TEXT PRIMARY KEY,
    last_fetch_ts INTEGER
);
"""

DAILY_COUNTER_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS twitter_daily_counter (
    day_key TEXT PRIMARY KEY,
    markets_fetched INTEGER
);
"""

GRANT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS twitter_daily_grants (
    day_key TEXT NOT NULL,
    market_id TEXT NOT NULL,
    PRIMARY KEY (day_key, market_id)
);
"""


class TwitterQuotaStore:
    """Lightweight SQLite helper to track X/Twitter quota and caching."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        if self.db_path.parent and not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute(QUOTA_TABLE_SQL)
        cur.execute(CACHE_TABLE_SQL)
        cur.execute(COOLDOWN_TABLE_SQL)
        cur.execute(DAILY_COUNTER_TABLE_SQL)
        cur.execute(GRANT_TABLE_SQL)
        self.conn.commit()

    # ------------------------------------------------------------------
    def get_quota_state(self, month_key: str, budget: int) -> Tuple[int, int]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT reads_used, budget FROM twitter_quota_state WHERE month_key = ?",
            (month_key,),
        )
        row = cur.fetchone()
        if row:
            if row["budget"] != budget:
                cur.execute(
                    "UPDATE twitter_quota_state SET budget = ? WHERE month_key = ?",
                    (budget, month_key),
                )
                self.conn.commit()
            return int(row["reads_used"] or 0), int(row["budget"] or budget)
        cur.execute(
            "INSERT INTO twitter_quota_state (month_key, reads_used, budget) VALUES (?, ?, ?)",
            (month_key, 0, budget),
        )
        self.conn.commit()
        return 0, budget

    def consume_quota(self, month_key: str, budget: int, amount: int = 1) -> None:
        used, _ = self.get_quota_state(month_key, budget)
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE twitter_quota_state SET reads_used = ? WHERE month_key = ?",
            (used + amount, month_key),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    def get_cache(self, cache_key: str, ttl_hours: int, now_ts: Optional[int] = None) -> Optional[dict]:
        now_ts = now_ts or int(time.time())
        cur = self.conn.cursor()
        cur.execute(
            "SELECT created_ts, response_json FROM twitter_query_cache WHERE cache_key = ?",
            (cache_key,),
        )
        row = cur.fetchone()
        if not row:
            return None
        created_ts = int(row["created_ts"])
        if now_ts - created_ts > ttl_hours * 3600:
            cur.execute("DELETE FROM twitter_query_cache WHERE cache_key = ?", (cache_key,))
            self.conn.commit()
            return None
        try:
            return json.loads(row["response_json"])
        except Exception:
            return None

    def set_cache(self, cache_key: str, response_json: dict, now_ts: Optional[int] = None) -> None:
        now_ts = now_ts or int(time.time())
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO twitter_query_cache (cache_key, created_ts, response_json)
            VALUES (?, ?, ?)
            """,
            (cache_key, now_ts, json.dumps(response_json)),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    def get_market_cooldown(self, market_id: str) -> Optional[int]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT last_fetch_ts FROM twitter_market_cooldown WHERE market_id = ?",
            (market_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return int(row["last_fetch_ts"])

    def update_market_cooldown(self, market_id: str, now_ts: Optional[int] = None) -> None:
        now_ts = now_ts or int(time.time())
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO twitter_market_cooldown (market_id, last_fetch_ts)
            VALUES (?, ?)
            """,
            (market_id, now_ts),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    def get_daily_count(self, day_key: str) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT markets_fetched FROM twitter_daily_counter WHERE day_key = ?",
            (day_key,),
        )
        row = cur.fetchone()
        return int(row["markets_fetched"]) if row else 0

    def increment_daily_count(self, day_key: str) -> None:
        current = self.get_daily_count(day_key)
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO twitter_daily_counter (day_key, markets_fetched)
            VALUES (?, ?)
            """,
            (day_key, current + 1),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    def get_daily_granted_count(self, day_key: str) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT COUNT(*) as cnt FROM twitter_daily_grants WHERE day_key = ?",
            (day_key,),
        )
        row = cur.fetchone()
        return int(row["cnt"] if row else 0)

    def get_granted_markets(self, day_key: str) -> set[str]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT market_id FROM twitter_daily_grants WHERE day_key = ?",
            (day_key,),
        )
        return {row["market_id"] for row in cur.fetchall()}

    def mark_market_granted(self, day_key: str, market_id: str) -> bool:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO twitter_daily_grants (day_key, market_id)
            VALUES (?, ?)
            """,
            (day_key, market_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.conn.close()
