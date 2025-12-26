"""Heuristic extraction of rules-only research features (no LLM needed)."""
from __future__ import annotations

import re
from typing import Dict

AMBIGUOUS_KEYWORDS = {
    "likely",
    "expected",
    "estimated",
    "could",
    "may",
    "might",
    "rumor",
    "unconfirmed",
    "as judged",
    "in our opinion",
    "to our satisfaction",
    "controversy",
    "disputed",
}

OBJECTIVE_KEYWORDS = {
    "official",
    "government",
    "court",
    "fec",
    "sec",
    "ecb",
    "fed",
    "cpi",
    "nfp",
    "announced",
    "published",
    "released",
    "certified",
    "gazette",
}

SOURCE_TYPE_KEYWORDS = {
    "official_statement": {"official", "statement", "press release"},
    "government_data": {"government", "cpi", "nfp", "gazette", "bureau", "labor"},
    "court_ruling": {"court", "ruling", "verdict", "lawsuit", "appeal"},
    "sports_result": {"game", "match", "tournament", "score", "season"},
    "onchain": {"on-chain", "onchain", "blockchain", "ethereum", "solana"},
}


def _clean_text(market: Dict) -> str:
    text_fields = [
        market.get("rules"),
        market.get("description"),
        market.get("question"),
        market.get("title"),
    ]
    text = " ".join([t for t in text_fields if t])
    return text.lower().strip()


def _has_keyword(text: str, keywords: set[str]) -> bool:
    return any(re.search(r"\b" + re.escape(word) + r"\b", text) for word in keywords)


def _source_type(text: str) -> str:
    for source_type, keywords in SOURCE_TYPE_KEYWORDS.items():
        if _has_keyword(text, keywords):
            return source_type
    if "official" in text:
        return "official_statement"
    return "unclear"


def extract_rules_features(market: Dict) -> Dict:
    """Derive research features from market rules/description alone."""

    text = _clean_text(market)
    is_short = len(text) < 32
    has_ambiguous_kw = _has_keyword(text, AMBIGUOUS_KEYWORDS)
    has_objective_kw = _has_keyword(text, OBJECTIVE_KEYWORDS)

    score = 0.0
    if is_short:
        score += 0.6
    if has_ambiguous_kw:
        score += 0.5
    if has_objective_kw:
        score -= 0.3

    score = max(0.0, min(1.0, score))
    rules_ambiguous = 1 if score >= 0.5 else 0

    return {
        "rules_ambiguous": rules_ambiguous,
        "resolution_source_type": _source_type(text) if text else "unclear",
        "ambiguity_score": score,
    }
