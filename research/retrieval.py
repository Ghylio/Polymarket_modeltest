"""Leakage-safe document retrieval for research features."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence

import numpy as np

from sentiment.store import DocumentStore


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return [t for t in tokens if t]


def _keyword_overlap_score(query_tokens: Sequence[str], doc: Dict) -> float:
    doc_text = " ".join(
        [
            str(doc.get("title") or ""),
            str(doc.get("text") or ""),
        ]
    ).lower()
    doc_tokens = _tokenize(doc_text)
    if not doc_tokens:
        return 0.0
    query_counts = Counter(query_tokens)
    doc_counts = Counter(doc_tokens)
    overlap = sum(min(query_counts[t], doc_counts[t]) for t in query_counts)
    return float(overlap) / float(len(doc_tokens) + 1e-6)


def _recency_score(published_ts: int, as_of_ts: int, window_hours: int) -> float:
    window_seconds = window_hours * 3600
    delta = max(0, as_of_ts - published_ts)
    if delta >= window_seconds:
        return 0.0
    return 1.0 - (delta / window_seconds)


def get_market_docs(
    store: DocumentStore,
    market: Dict,
    as_of_ts: int,
    window_hours: int,
    top_k: int = 20,
    use_embeddings: bool = False,
) -> List[Dict]:
    """Return top-k docs for a market without leakage.

    Documents are filtered to published_ts <= as_of_ts and within the
    requested window. Ranking uses a blend of recency and keyword overlap with
    the market question/title. If embeddings are present and enabled, a cosine
    similarity score is blended in as well.
    """

    start_ts = as_of_ts - window_hours * 3600
    docs = store.fetch_docs(market_id=market.get("id"), start_ts=start_ts, end_ts=as_of_ts)
    if not docs:
        return []

    title = (market.get("question") or market.get("title") or "").lower()
    desc = (market.get("description") or market.get("rules") or "")
    query_tokens = _tokenize(title + " " + desc)

    scored: List[Dict] = []
    for row in docs:
        doc = dict(row)
        published_ts = int(doc.get("published_ts") or 0)
        if published_ts <= 0 or published_ts > as_of_ts:
            continue
        overlap = _keyword_overlap_score(query_tokens, doc) if query_tokens else 0.0
        recency = _recency_score(published_ts, as_of_ts, window_hours)
        embed_score = 0.0
        if use_embeddings:
            embed_vec = doc.get("embedding")
            market_vec = market.get("embedding")
            if isinstance(embed_vec, (list, tuple)) and isinstance(market_vec, (list, tuple)):
                embed_score = _cosine_similarity(embed_vec, market_vec)
        total_score = 0.65 * recency + 0.35 * overlap + 0.2 * embed_score
        doc["_score"] = total_score
        scored.append(doc)

    scored.sort(key=lambda d: d.get("_score", 0.0), reverse=True)
    return scored[:top_k]


def _cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    a_arr = np.array(list(a), dtype=float)
    b_arr = np.array(list(b), dtype=float)
    if a_arr.size == 0 or b_arr.size == 0:
        return 0.0
    denom = math.sqrt(float((a_arr**2).sum())) * math.sqrt(float((b_arr**2).sum()))
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)
