"""Sentiment provider configuration loader."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = Path("config/sentiment_config.yaml")


def load_sentiment_config(path: Path | None = None) -> Dict[str, Any]:
    """Load sentiment config with safe defaults.

    If the file is missing or invalid, fall back to an empty config where
    providers are disabled to avoid hard failures.
    """

    cfg_path = path or DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        logging.getLogger(__name__).info("Sentiment config not found at %s; using defaults", cfg_path)
        return {
            "providers": {
                "gdelt": {"enabled": False},
                "newsapi": {"enabled": False, "api_key": ""},
                "twitter": {"enabled": False, "bearer_token": ""},
            },
            "sentiment": {"enabled": False, "model": None},
        }

    try:
        with cfg_path.open("r") as f:
            loaded = yaml.safe_load(f) or {}
    except Exception as exc:  # pragma: no cover - defensive
        logging.getLogger(__name__).warning("Failed to load sentiment config: %s", exc)
        return {
            "providers": {
                "gdelt": {"enabled": False},
                "newsapi": {"enabled": False, "api_key": ""},
                "twitter": {"enabled": False, "bearer_token": ""},
            },
            "sentiment": {"enabled": False, "model": None},
        }

    providers = loaded.get("providers") or {}
    sentiment = loaded.get("sentiment") or {}

    return {
        "providers": {
            "gdelt": providers.get("gdelt") or {"enabled": False},
            "newsapi": providers.get("newsapi") or {"enabled": False, "api_key": ""},
            "twitter": providers.get("twitter") or {"enabled": False, "bearer_token": ""},
        },
        "sentiment": {
            "enabled": bool(sentiment.get("enabled", False)),
            "model": sentiment.get("model"),
        },
    }
