"""Pluggable sentiment news providers."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
import hashlib
import time
from typing import Dict, Iterable, List, Optional, Tuple

import requests

from sentiment.quota_store import TwitterQuotaStore

LOGGER = logging.getLogger(__name__)


class SentimentProvider(ABC):
    name: str = "base"

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    @abstractmethod
    def fetch(
        self, query: str, start_time: datetime, end_time: datetime, **kwargs
    ) -> List[Dict[str, str]]:
        """Return list of documents with keys: text, published_at."""
        raise NotImplementedError


class GDELTProvider(SentimentProvider):
    name = "gdelt"
    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def fetch(
        self, query: str, start_time: datetime, end_time: datetime, **kwargs
    ) -> List[Dict[str, str]]:
        if not self.enabled:
            return []

        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": 50,
            "format": "json",
            "startdatetime": start_time.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end_time.strftime("%Y%m%d%H%M%S"),
        }
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles") or []
            return [
                {"text": art.get("title") or art.get("seendate", ""), "published_at": art.get("seendate")}
                for art in articles
            ]
        except Exception as exc:  # pragma: no cover - network path
            LOGGER.warning("GDELT provider failed: %s", exc)
            return []


class NewsAPIProvider(SentimentProvider):
    name = "newsapi"
    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: Optional[str], enabled: bool = True):
        super().__init__(enabled=enabled and bool(api_key))
        self.api_key = api_key

    def fetch(
        self, query: str, start_time: datetime, end_time: datetime, **kwargs
    ) -> List[Dict[str, str]]:
        if not self.enabled:
            return []

        params = {
            "q": query,
            "from": start_time.isoformat(),
            "to": end_time.isoformat(),
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 50,
            "apiKey": self.api_key,
        }
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles") or []
            return [
                {
                    "text": (art.get("title") or "") + " " + (art.get("description") or ""),
                    "published_at": art.get("publishedAt"),
                }
                for art in articles
            ]
        except Exception as exc:  # pragma: no cover - network path
            LOGGER.warning("NewsAPI provider failed: %s", exc)
            return []


class TwitterProvider(SentimentProvider):
    name = "twitter"
    BASE_URL = "https://api.twitter.com/2/tweets/search/recent"

    DEFAULTS = {
        "monthly_read_budget": 80,
        "max_markets_per_day": 3,
        "query_cache_ttl_hours": 24,
        "cooldown_hours_per_market": 24,
        "max_results_per_query": 10,
        "min_other_docs_24h": 3,
        "lang": "en",
        "exclude_retweets": True,
    }

    def __init__(
        self,
        bearer_token: Optional[str],
        enabled: bool = True,
        config: Optional[Dict] = None,
        quota_db_path: str | None = None,
    ):
        super().__init__(enabled=enabled and bool(bearer_token))
        self.bearer_token = bearer_token
        self.config = {**self.DEFAULTS, **(config or {})}
        self.quota_store = TwitterQuotaStore(quota_db_path or "data/twitter_quota.db")
        self._quota_warned = False

    def _cache_key(self, query: str, now_ts: int) -> str:
        day_key = datetime.utcfromtimestamp(now_ts).strftime("%Y-%m-%d")
        return hashlib.sha1(f"{query}-{day_key}".encode("utf-8")).hexdigest()

    def _month_key(self, now_ts: int) -> str:
        return datetime.utcfromtimestamp(now_ts).strftime("%Y-%m")

    def _day_key(self, now_ts: int) -> str:
        return datetime.utcfromtimestamp(now_ts).strftime("%Y-%m-%d")

    def _prepare_query(self, query: str) -> str:
        parts = [query]
        if self.config.get("exclude_retweets"):
            parts.append("-is:retweet")
        if self.config.get("lang"):
            parts.append(f"lang:{self.config['lang']}")
        return " ".join(parts)

    def _from_cache(self, cached_json: dict) -> List[Dict[str, str]]:
        tweets = cached_json.get("data") or []
        return [
            {"text": tw.get("text", ""), "published_at": tw.get("created_at")}
            for tw in tweets
        ]

    def should_fetch(
        self,
        market_id: Optional[str],
        query: str,
        other_docs_24h_count: Optional[int] = None,
        now_ts: Optional[int] = None,
    ) -> Tuple[bool, str, Optional[List[Dict[str, str]]]]:
        now_ts = now_ts or int(time.time())
        if not self.enabled:
            return False, "disabled", None
        if not self.bearer_token:
            return False, "no_token", None

        other_docs = other_docs_24h_count or 0
        if other_docs >= self.config.get("min_other_docs_24h", 0):
            return False, "sufficient_other_coverage", None

        month_key = self._month_key(now_ts)
        used, budget = self.quota_store.get_quota_state(
            month_key, self.config.get("monthly_read_budget", self.DEFAULTS["monthly_read_budget"])
        )
        if used >= budget:
            if not self._quota_warned:
                LOGGER.warning("Twitter monthly budget exhausted for %s (used=%s, budget=%s)", month_key, used, budget)
                self._quota_warned = True
            return False, "quota_exhausted", None

        day_key = self._day_key(now_ts)
        if self.quota_store.get_daily_count(day_key) >= self.config.get("max_markets_per_day", self.DEFAULTS["max_markets_per_day"]):
            return False, "day_cap", None

        if market_id:
            last_fetch = self.quota_store.get_market_cooldown(market_id)
            cooldown_sec = int(self.config.get("cooldown_hours_per_market", self.DEFAULTS["cooldown_hours_per_market"]) * 3600)
            if last_fetch and now_ts - last_fetch < cooldown_sec:
                return False, "cooldown", None

        cache_key = self._cache_key(query, now_ts)
        cached = self.quota_store.get_cache(
            cache_key,
            ttl_hours=int(self.config.get("query_cache_ttl_hours", self.DEFAULTS["query_cache_ttl_hours"])),
            now_ts=now_ts,
        )
        if cached:
            return False, "cache_hit", self._from_cache(cached)

        return True, "ok", None

    def _log_metric(self, event: str, payload: Dict) -> None:
        LOGGER.info("%s: %s", event, payload)

    def fetch(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        market_id: Optional[str] = None,
        other_docs_24h_count: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        now_ts = int(time.time())
        prepared_query = self._prepare_query(query)
        should, reason, cached_docs = self.should_fetch(
            market_id=market_id,
            query=prepared_query,
            other_docs_24h_count=other_docs_24h_count,
            now_ts=now_ts,
        )
        if not should:
            self._log_metric("twitter_fetch_skip", {"reason": reason, "market_id": market_id})
            if reason == "cache_hit" and cached_docs is not None:
                self._log_metric("twitter_fetch_ok", {"market_id": market_id, "count": len(cached_docs), "cache_hit": True})
                return cached_docs
            return []

        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        max_results = int(self.config.get("max_results_per_query", self.DEFAULTS["max_results_per_query"]))
        max_results = max(10, min(max_results, 100))
        params = {
            "query": prepared_query,
            "start_time": start_time.isoformat(timespec="seconds") + "Z",
            "end_time": end_time.isoformat(timespec="seconds") + "Z",
            "max_results": max_results,
            "tweet.fields": "created_at,lang",
        }
        try:
            resp = requests.get(self.BASE_URL, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            tweets = data.get("data") or []
            docs = [
                {"text": tw.get("text", ""), "published_at": tw.get("created_at")}
                for tw in tweets
            ]
            month_key = self._month_key(now_ts)
            self.quota_store.consume_quota(
                month_key,
                self.config.get("monthly_read_budget", self.DEFAULTS["monthly_read_budget"]),
            )
            self.quota_store.increment_daily_count(self._day_key(now_ts))
            if market_id:
                self.quota_store.update_market_cooldown(market_id, now_ts=now_ts)
            cache_key = self._cache_key(prepared_query, now_ts)
            self.quota_store.set_cache(cache_key, data, now_ts=now_ts)
            self._log_metric(
                "twitter_fetch_ok",
                {"market_id": market_id, "count": len(docs), "cache_hit": False},
            )
            self._log_metric(
                "twitter_quota_state",
                {
                    "month_key": month_key,
                    "used": self.quota_store.get_quota_state(
                        month_key, self.config.get("monthly_read_budget", self.DEFAULTS["monthly_read_budget"])
                    )[0],
                    "budget": self.config.get("monthly_read_budget", self.DEFAULTS["monthly_read_budget"]),
                },
            )
            return docs
        except Exception as exc:  # pragma: no cover - network path
            LOGGER.warning("Twitter provider failed: %s", exc)
            return []


def build_providers_from_config(config: Dict[str, Dict]) -> List[SentimentProvider]:
    providers_cfg = config.get("providers") or {}
    providers: List[SentimentProvider] = []

    gdelt_cfg = providers_cfg.get("gdelt") or {"enabled": False}
    providers.append(GDELTProvider(enabled=bool(gdelt_cfg.get("enabled", False))))

    news_cfg = providers_cfg.get("newsapi") or {}
    providers.append(
        NewsAPIProvider(
            api_key=news_cfg.get("api_key"),
            enabled=bool(news_cfg.get("enabled", False)),
        )
    )

    twitter_cfg = providers_cfg.get("twitter") or {}
    providers.append(
        TwitterProvider(
            bearer_token=twitter_cfg.get("bearer_token"),
            enabled=bool(twitter_cfg.get("enabled", False)),
            config=twitter_cfg,
            quota_db_path=twitter_cfg.get("quota_db_path"),
        )
    )

    # Filter out disabled providers to simplify downstream loops
    return [p for p in providers if p.enabled]
