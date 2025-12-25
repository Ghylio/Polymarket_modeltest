"""Pluggable sentiment news providers."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import requests

LOGGER = logging.getLogger(__name__)


class SentimentProvider(ABC):
    name: str = "base"

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    @abstractmethod
    def fetch(
        self, query: str, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, str]]:
        """Return list of documents with keys: text, published_at."""
        raise NotImplementedError


class GDELTProvider(SentimentProvider):
    name = "gdelt"
    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def fetch(
        self, query: str, start_time: datetime, end_time: datetime
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
        self, query: str, start_time: datetime, end_time: datetime
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

    def __init__(self, bearer_token: Optional[str], enabled: bool = True):
        super().__init__(enabled=enabled and bool(bearer_token))
        self.bearer_token = bearer_token

    def fetch(
        self, query: str, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, str]]:
        if not self.enabled:
            return []

        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        params = {
            "query": query,
            "start_time": start_time.isoformat(timespec="seconds") + "Z",
            "end_time": end_time.isoformat(timespec="seconds") + "Z",
            "max_results": 50,
            "tweet.fields": "created_at,text",
        }
        try:
            resp = requests.get(self.BASE_URL, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            tweets = data.get("data") or []
            return [
                {"text": tw.get("text", ""), "published_at": tw.get("created_at")}
                for tw in tweets
            ]
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
        )
    )

    # Filter out disabled providers to simplify downstream loops
    return [p for p in providers if p.enabled]
