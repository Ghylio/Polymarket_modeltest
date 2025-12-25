"""Sentiment ingestion and aggregation."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

import numpy as np

from data.sentiment_providers import SentimentProvider

LOGGER = logging.getLogger(__name__)

STOPWORDS = {
    "the",
    "a",
    "an",
    "to",
    "for",
    "of",
    "and",
    "or",
    "in",
    "on",
    "at",
    "by",
    "will",
    "is",
    "be",
    "vs",
}
ALIASES = {"u.s.": "united states", "usa": "united states", "uk": "united kingdom"}

SENTIMENT_WINDOWS = ["1h", "6h", "24h", "7d"]
SENTIMENT_ALIAS_MAP = {
    "1d": "24h",
    "24hr": "24h",
    "7day": "7d",
}
SENTIMENT_COLUMNS = [
    f"sent_mean_{w}" for w in SENTIMENT_WINDOWS
] + [
    f"sent_std_{w}" for w in SENTIMENT_WINDOWS
] + [
    f"doc_count_{w}" for w in SENTIMENT_WINDOWS
] + ["sent_trend"]


def canonicalize_sentiment_features(features: Dict[str, float]) -> Dict[str, float]:
    """Ensure all canonical sentiment columns exist, filling with NaN when missing."""

    canonical = {}
    for key, value in list(features.items()):
        for alias, target in SENTIMENT_ALIAS_MAP.items():
            if key.endswith(alias):
                new_key = key.replace(alias, target)
                canonical[new_key] = value
    canonical.update(features)

    for col in SENTIMENT_COLUMNS:
        if col not in canonical:
            canonical[col] = np.nan
    return canonical


class SentimentScorer:
    """Wrapper around a transformer sentiment pipeline."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or "distilbert-base-uncased-finetuned-sst-2-english"
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline  # type: ignore

                self._pipeline = pipeline("sentiment-analysis", model=self.model_name)
            except Exception as exc:  # pragma: no cover - import heavy path
                LOGGER.warning("Sentiment model unavailable: %s", exc)
                self._pipeline = None

    def score(self, text: str) -> float:
        self._ensure_pipeline()
        if not self._pipeline:
            return 0.0
        try:
            result = self._pipeline(text[:512])[0]
            if isinstance(result, dict):
                label = result.get("label", "").lower()
                score = float(result.get("score", 0.0))
                return score if "pos" in label else 1 - score
        except Exception as exc:  # pragma: no cover - runtime path
            LOGGER.warning("Sentiment scoring failed: %s", exc)
        return 0.0


class SentimentFeatureBuilder:
    def __init__(
        self,
        providers: Iterable[SentimentProvider],
        scorer: SentimentScorer,
        windows: Optional[List[timedelta]] = None,
        enabled: bool = True,
    ):
        self.providers = list(providers)
        self.scorer = scorer
        self.enabled = enabled
        self.windows = windows or [
            timedelta(hours=1),
            timedelta(hours=6),
            timedelta(hours=24),
            timedelta(days=7),
        ]

    # --------------------------------------------------------------
    def generate_queries(self, market: Dict) -> List[str]:
        title = (market.get("question") or market.get("title") or "").lower()
        desc = (market.get("description") or "").lower()
        tokens = [t.strip(" ?,.:;!()") for t in (title + " " + desc).split()]
        terms = []
        for tok in tokens:
            if not tok or tok in STOPWORDS:
                continue
            terms.append(ALIASES.get(tok, tok))
        unique_terms = list(dict.fromkeys(terms))
        if not unique_terms:
            return []
        queries = [" ".join(unique_terms[:6])]
        if len(unique_terms) > 6:
            queries.append(" ".join(unique_terms[6:12]))
        return queries

    def build_features(
        self, market: Dict, as_of: datetime
    ) -> Dict[str, float]:
        if not self.enabled:
            return canonicalize_sentiment_features({})

        queries = self.generate_queries(market)
        if not queries:
            return canonicalize_sentiment_features({})

        window_features: Dict[str, Dict[str, float]] = {}
        for window in self.windows:
            bucket_label = self._label(window)
            agg = self._aggregate_for_window(queries, as_of, window)
            if agg is None:
                continue
            window_features[bucket_label] = agg

        features: Dict[str, float] = {}
        for label in SENTIMENT_WINDOWS:
            agg = window_features.get(label)
            features[f"sent_mean_{label}"] = agg.get("mean", np.nan) if agg else np.nan
            features[f"sent_std_{label}"] = agg.get("std", np.nan) if agg else np.nan
            features[f"doc_count_{label}"] = agg.get("count", np.nan) if agg else np.nan

        if "1h" in window_features and "24h" in window_features:
            features["sent_trend"] = window_features["1h"].get("mean", np.nan) - window_features["24h"].get("mean", np.nan)
        else:
            features["sent_trend"] = np.nan

        return canonicalize_sentiment_features(features)

    # --------------------------------------------------------------
    def _aggregate_for_window(
        self, queries: List[str], as_of: datetime, window: timedelta
    ) -> Optional[Dict[str, float]]:
        start_time = as_of - window
        docs: List[str] = []
        for provider in self.providers:
            for query in queries:
                try:
                    articles = provider.fetch(query, start_time=start_time, end_time=as_of)
                    docs.extend([art.get("text", "") for art in articles if art.get("text")])
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Provider %s failed: %s", provider.name, exc)
                    continue
        if not docs:
            return None

        scores = [self.scorer.score(doc) for doc in docs]
        if not scores:
            return None

        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)) if len(scores) > 1 else 0.0,
            "count": float(len(scores)),
        }

    def _label(self, window: timedelta) -> str:
        hours = int(window.total_seconds() // 3600)
        if hours <= 24:
            return f"{hours}h"
        days = int(hours // 24)
        return f"{days}d"
