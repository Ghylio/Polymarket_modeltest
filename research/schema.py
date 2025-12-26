from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from typing import Any, Dict, Mapping, Optional

RESEARCH_COLUMNS = [
    "llm_p_yes",
    "llm_confidence",
    "evidence_count_24h",
    "evidence_count_7d",
    "source_diversity_7d",
    "stance_score_24h",
    "stance_score_7d",
    "rules_ambiguous",
    "resolution_source_type",
    "ambiguity_score",
]

COUNT_COLUMNS = {"evidence_count_24h", "evidence_count_7d", "source_diversity_7d"}
BOOLEAN_COLUMNS = {"rules_ambiguous"}
STRING_COLUMNS = {"resolution_source_type"}
FLOAT_COLUMNS = {"ambiguity_score"}


class ResearchFeatures(BaseModel):
    llm_p_yes: Optional[float] = Field(default=None)
    llm_confidence: Optional[float] = Field(default=None)
    evidence_count_24h: Optional[int] = Field(default=None)
    evidence_count_7d: Optional[int] = Field(default=None)
    source_diversity_7d: Optional[int] = Field(default=None)
    stance_score_24h: Optional[float] = Field(default=None)
    stance_score_7d: Optional[float] = Field(default=None)
    rules_ambiguous: Optional[int] = Field(default=None)
    resolution_source_type: Optional[str] = Field(default=None)
    ambiguity_score: Optional[float] = Field(default=None)

    def canonical(self) -> Dict[str, Any]:
        return canonicalize_research_features(self.dict())


def canonicalize_research_features(features: Mapping[str, Any]) -> Dict[str, Any]:
    canonical: Dict[str, Any] = {}
    canonical.update({k: v for k, v in features.items() if k in RESEARCH_COLUMNS})

    defaults = default_research_features()
    for col in RESEARCH_COLUMNS:
        value = canonical.get(col, defaults[col])
        if col in COUNT_COLUMNS or col in BOOLEAN_COLUMNS:
            canonical[col] = int(value) if value is not None and not _is_nan(value) else 0
        elif col in STRING_COLUMNS:
            canonical[col] = value if value not in (None, "") else np.nan
        elif col in FLOAT_COLUMNS:
            canonical[col] = float(value) if value is not None and not _is_nan(value) else np.nan
        else:
            canonical[col] = float(value) if value is not None and not _is_nan(value) else np.nan

    return canonical


def default_research_features() -> Dict[str, Any]:
    return {
        "llm_p_yes": np.nan,
        "llm_confidence": np.nan,
        "evidence_count_24h": 0,
        "evidence_count_7d": 0,
        "source_diversity_7d": 0,
        "stance_score_24h": np.nan,
        "stance_score_7d": np.nan,
        "rules_ambiguous": 0,
        "resolution_source_type": np.nan,
        "ambiguity_score": np.nan,
    }


def _is_nan(value: Any) -> bool:
    try:
        return bool(np.isnan(value))
    except Exception:
        return False
