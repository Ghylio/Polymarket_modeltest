"""Shared market quality filters for trading and backtesting."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class MarketFilterConfig:
    min_24h_volume: float = 0.0
    max_spread_abs: Optional[float] = 0.1
    max_spread_pct: Optional[float] = 0.25
    min_top_of_book_depth: float = 0.0
    min_trades_last_24h: int = 0
    skip_if_rules_ambiguous: bool = True
    rules_min_length: int = 40
    ambiguous_keywords: Iterable[str] = field(
        default_factory=lambda: ["subjective", "opinion", "criteria", "discretion"]
    )
    skip_if_time_to_resolve_lt: float = 10 * 60  # seconds


def _extract(market_state: Dict, key: str, default=None):
    if hasattr(market_state, key):
        return getattr(market_state, key)
    return market_state.get(key, default)


def should_trade_market(
    market_state: Dict,
    market_metadata: Dict,
    now_ts: Optional[float] = None,
    config: Optional[MarketFilterConfig] = None,
) -> Tuple[bool, List[str]]:
    """Evaluate whether a market passes quality filters.

    Args:
        market_state: dict-like with bid/ask/mid/spread/depth/volume/trades counts
        market_metadata: raw market payload
        now_ts: optional override for current timestamp
        config: MarketFilterConfig overrides

    Returns:
        (should_trade, reasons)
    """

    cfg = config or MarketFilterConfig()
    reasons: List[str] = []
    now = now_ts or time.time()

    # Volume filter
    vol_keys = ["volume24hr", "volume24h", "volume", "totalVolume"]
    volume = _extract(market_state, "volume24h")
    if volume is None:
        for key in vol_keys:
            if key in market_metadata:
                try:
                    volume = float(market_metadata.get(key))
                    break
                except Exception:
                    continue
    if cfg.min_24h_volume > 0:
        if volume is None or math.isnan(volume):
            reasons.append("missing_volume")
        elif volume < cfg.min_24h_volume:
            reasons.append("low_volume")

    # Spread filter
    bid = _extract(market_state, "bid", math.nan)
    ask = _extract(market_state, "ask", math.nan)
    spread = _extract(market_state, "spread", math.nan)
    mid = _extract(market_state, "mid", 0.5)
    if math.isnan(spread) and not (math.isnan(bid) or math.isnan(ask)):
        spread = abs(ask - bid)
    if cfg.max_spread_abs is not None and not math.isnan(spread):
        if spread > cfg.max_spread_abs:
            reasons.append("wide_spread_abs")
    if cfg.max_spread_pct is not None and not math.isnan(spread):
        denom = max(mid, 1e-6)
        if spread / denom > cfg.max_spread_pct:
            reasons.append("wide_spread_pct")

    # Depth filter
    depth = _extract(market_state, "depth")
    if cfg.min_top_of_book_depth > 0:
        if depth is None or math.isnan(depth):
            reasons.append("missing_depth")
        elif depth < cfg.min_top_of_book_depth:
            reasons.append("shallow_book")

    # Trades activity filter
    trades_24h = _extract(market_state, "trades_last_24h")
    if cfg.min_trades_last_24h > 0:
        if trades_24h is None:
            reasons.append("missing_trades")
        elif trades_24h < cfg.min_trades_last_24h:
            reasons.append("inactive_trades")

    # Rule ambiguity filter
    if cfg.skip_if_rules_ambiguous:
        rules_text = str(market_metadata.get("rules") or market_metadata.get("description") or "")
        if len(rules_text.strip()) < cfg.rules_min_length:
            reasons.append("short_rules")
        lower_rules = rules_text.lower()
        if any(keyword in lower_rules for keyword in cfg.ambiguous_keywords):
            reasons.append("ambiguous_rules")

    # Time to resolve filter
    if cfg.skip_if_time_to_resolve_lt:
        time_to_resolve = _extract(market_state, "time_to_resolve")
        if time_to_resolve is None:
            # Try metadata timestamps
            ts_keys = ["endTime", "endDate", "closeTime", "resolutionTime"]
            for key in ts_keys:
                ts_val = market_metadata.get(key)
                if ts_val:
                    try:
                        end_ts = float(ts_val)
                        # If ts is likely ms, convert
                        if end_ts > 1e12:
                            end_ts = end_ts / 1000.0
                        time_to_resolve = end_ts - now
                        break
                    except Exception:
                        continue
        if time_to_resolve is not None:
            if time_to_resolve < cfg.skip_if_time_to_resolve_lt:
                reasons.append("too_close_to_resolution")

    return len(reasons) == 0, reasons

