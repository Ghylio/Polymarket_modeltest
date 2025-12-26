"""Prompt construction and parsing for research feature LLM calls."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List

from research.schema import RESEARCH_COLUMNS, ResearchFeatures


def build_prompt(market: Dict, docs: List[Dict], as_of_ts: int) -> str:
    """Construct a strict JSON-only prompt for the LLM."""

    market_title = market.get("question") or market.get("title") or ""
    market_rules = market.get("rules") or market.get("description") or ""

    doc_lines = []
    for doc in docs:
        ts = doc.get("published_ts") or 0
        ts_str = datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
        snippet = (doc.get("text") or doc.get("title") or "").strip().replace("\n", " ")
        snippet = snippet[:280]
        doc_lines.append(
            f"- {ts_str} | {doc.get('source') or doc.get('provider') or 'unknown'} | "
            f"{doc.get('title') or ''} :: {snippet}"
        )

    schema_example = {
        "llm_p_yes": 0.42,
        "llm_confidence": 0.61,
        "evidence_count_24h": 3,
        "evidence_count_7d": 7,
        "source_diversity_7d": 3,
        "stance_score_24h": -0.2,
        "stance_score_7d": 0.1,
        "rules_ambiguous": 0,
        "resolution_source_type": "official",
    }

    prompt = f"""
You are an analyst. Given a market question and recent documents, respond with STRICT JSON ONLY.
Do not include any prose or markdown. If unsure for a field, output null.

Current time (UTC): {datetime.utcfromtimestamp(as_of_ts).isoformat()}Z
Market question: {market_title}
Rules/description: {market_rules}

Top documents (most relevant first):
{chr(10).join(doc_lines) if doc_lines else 'None'}

Return a single JSON object exactly matching this schema:
{json.dumps(schema_example, indent=2)}

Only return JSON. No commentary.
"""
    return prompt.strip()


def parse_llm_output(raw: str | Dict) -> ResearchFeatures:
    """Parse LLM output into the typed ResearchFeatures schema."""

    if isinstance(raw, dict):
        data = raw
    else:
        try:
            data = json.loads(raw)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid JSON from LLM: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("LLM output must be a JSON object")

    filtered = {k: data.get(k) for k in RESEARCH_COLUMNS if k in data}
    return ResearchFeatures(**filtered)
