"""Subgraph configuration loader."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = Path("config/subgraph_config.yaml")


_DEF_CFG = {
    "subgraph": {
        "enabled": False,
        "url": "https://api.thegraph.com/subgraphs/name/polymarket/matic-markets",
        "resolution_url": "https://api.thegraph.com/subgraphs/name/polymarket/resolutions",
        "volume_lookback_hours": 24,
    }
}


def load_subgraph_config(path: Path | None = None) -> Dict[str, Any]:
    """Load subgraph configuration with safe defaults."""

    cfg_path = path or DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        logging.getLogger(__name__).info(
            "Subgraph config not found at %s; using defaults", cfg_path
        )
        return _DEF_CFG

    try:
        with cfg_path.open("r") as f:
            loaded = yaml.safe_load(f) or {}
    except Exception as exc:  # pragma: no cover - defensive
        logging.getLogger(__name__).warning("Failed to load subgraph config: %s", exc)
        return _DEF_CFG

    subgraph_cfg = loaded.get("subgraph") or {}
    return {
        "subgraph": {
            "enabled": bool(subgraph_cfg.get("enabled", False)),
            "url": subgraph_cfg.get("url") or _DEF_CFG["subgraph"]["url"],
            "resolution_url": subgraph_cfg.get("resolution_url")
            or _DEF_CFG["subgraph"]["resolution_url"],
            "volume_lookback_hours": int(
                subgraph_cfg.get(
                    "volume_lookback_hours", _DEF_CFG["subgraph"]["volume_lookback_hours"]
                )
            ),
        }
    }
