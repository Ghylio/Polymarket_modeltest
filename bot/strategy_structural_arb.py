"""Structural arbitrage strategy (sum-to-1 Dutch book detection)."""

from __future__ import annotations

import time
import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from bot.market_filters import MarketFilterConfig, should_trade_market
from metrics import MetricsLogger
from negrisk_arb import NegRiskArbitrageur, OrderExecutor, PaperBatchExecutor
from polymarket_fetcher import PolymarketFetcher

if TYPE_CHECKING:  # pragma: no cover
    from trading_bot import RiskManager


TICK_SIZE = 0.01


@dataclass
class StructuralArbConfig:
    """Typed configuration for the structural arbitrage strategy.

    New callers should prefer the `enabled` / `min_edge_abs` style fields. Legacy
    properties (``arb_*``) are preserved for runtime compatibility.
    """

    enabled: bool = False
    allow_non_negrisk: bool = False
    max_legs_per_event: int = 15
    max_outcomes: int = 15

    min_edge_abs: float = 0.02
    slippage_bps: float = 10.0
    fee_bps: float = 0.0

    min_depth_per_leg_usdc: float = 250.0
    max_usdc_per_event: float = 2000.0
    max_usdc_per_cycle: float = 500.0

    use_fok_or_ioc: bool = True
    cooldown_after_partial_fill_sec: float = 300.0
    max_attempt_loss_per_day: float = 50.0

    max_batches_per_minute: int = 20

    # Depth robustness
    depth_robustness_enabled: bool = True
    depth_robustness_ticks: int = 1
    depth_robustness_max_extra_cost_abs: float = 0.01
    min_depth_within_band_usdc: float = 250.0

    def __post_init__(self) -> None:
        if self.max_legs_per_event > 15:
            original = self.max_legs_per_event
            self.max_legs_per_event = 15
            logging.getLogger(__name__).warning(
                "max_legs_per_event %s exceeds batch limit; clamping to 15", original
            )

    # ------------------------------------------------------------------
    # Helpers / legacy aliases
    # ------------------------------------------------------------------
    def edge_buffer(self) -> float:
        """Return the aggregate edge buffer implied by slippage and fees."""

        return (self.slippage_bps + self.fee_bps) / 10000.0

    # Legacy accessors used by existing strategy logic -----------------
    @property
    def arb_enabled(self) -> bool:
        return self.enabled

    @property
    def arb_edge_min(self) -> float:
        return self.min_edge_abs

    @property
    def arb_slippage_buffer(self) -> float:
        return self.edge_buffer()

    @property
    def arb_execution_buffer(self) -> float:
        return 0.0

    @property
    def max_arb_notional_per_event(self) -> float:
        return self.max_usdc_per_event

    @property
    def max_arb_attempt_loss_per_day(self) -> float:
        return self.max_attempt_loss_per_day

    @property
    def arb_max_outcomes(self) -> int:
        return self.max_outcomes

    @property
    def arb_min_depth_per_leg(self) -> float:
        return self.min_depth_per_leg_usdc

    @property
    def arb_prefer_neg_risk_only(self) -> bool:
        return not self.allow_non_negrisk


class StructuralArbStrategy:
    """Detect and trade full-set structural arbitrage opportunities."""

    def __init__(
        self,
        fetcher: Optional[PolymarketFetcher] = None,
        executor: Optional[OrderExecutor] = None,
        risk_manager: Optional["RiskManager"] = None,
        config: Optional[StructuralArbConfig] = None,
        filter_config: Optional[MarketFilterConfig] = None,
        metrics_logger: Optional[MetricsLogger] = None,
    ):
        self.fetcher = fetcher or PolymarketFetcher(verbose=False)
        self.executor = executor or PaperBatchExecutor()
        if risk_manager is None:
            from trading_bot import RiskManager

            self.risk = RiskManager()
        else:
            self.risk = risk_manager
        self.config = config or StructuralArbConfig()
        self.filter_config = filter_config or MarketFilterConfig()
        self.metrics_logger = metrics_logger
        self.cooldowns: Dict[str, float] = {}
        self._batch_times: deque[float] = deque()
        self._cycle_notional: float = 0.0

    def reset_cycle(self) -> None:
        """Reset per-cycle accounting for USDC spend limits."""

        self._cycle_notional = 0.0

    # ------------------------------------------------------------------
    def _collect_tokens_for_group(self, markets: Sequence[Dict]) -> List[Tuple[str, Dict]]:
        legs: List[Tuple[str, Dict]] = []
        for market in markets:
            mapping = self.fetcher.get_outcome_token_map(market)
            if not mapping:
                continue
            for outcome, token_id in (mapping.get("outcome_token_map") or {}).items():
                legs.append((token_id, market))
        return legs

    def _parse_asks(self, asks: Sequence) -> List[Tuple[float, float]]:
        parsed: List[Tuple[float, float]] = []
        for entry in asks:
            if isinstance(entry, dict):
                price = float(entry.get("price") or entry.get("[0]", np.nan))
                size = float(entry.get("size") or entry.get("[1]", np.nan))
            else:
                price = float(entry[0])
                size = float(entry[1]) if len(entry) > 1 else np.nan
            if not np.isnan(price) and not np.isnan(size):
                parsed.append((price, size))
        parsed.sort(key=lambda x: x[0])
        return parsed

    def _robust_depth(self, token_id: str) -> Tuple[Optional[float], Optional[float], float, float, float]:
        """Return best ask, best size, robust size, robust notional, and robust VWAP."""

        ob = self.fetcher.get_orderbook(token_id, depth=5)
        asks_raw = ob.get("asks", []) if ob else []
        asks = self._parse_asks(asks_raw)
        if not asks:
            return None, None, 0.0, 0.0, np.nan

        best_price, best_size = asks[0]
        band = self.config.depth_robustness_ticks * TICK_SIZE
        band = min(band, self.config.depth_robustness_max_extra_cost_abs)
        band_price = best_price + max(band, 0.0)

        robust_size = 0.0
        robust_notional = 0.0
        for price, size in asks:
            if price <= band_price:
                robust_size += size
                robust_notional += price * size
            else:
                break

        robust_vwap = robust_notional / robust_size if robust_size > 0 else np.nan
        return best_price, best_size, robust_size, robust_notional, robust_vwap

    def _within_batch_limit(self) -> bool:
        now = time.time()
        window_start = now - 60.0
        while self._batch_times and self._batch_times[0] < window_start:
            self._batch_times.popleft()
        if self.config.max_batches_per_minute and len(self._batch_times) >= self.config.max_batches_per_minute:
            return False
        self._batch_times.append(now)
        return True

    def _should_skip_group(self, event_id: str) -> bool:
        cooldown_until = self.cooldowns.get(event_id, 0)
        if cooldown_until and cooldown_until > time.time():
            if self.metrics_logger:
                self.metrics_logger.log_event(
                    "arb_skip", payload={"event_id": event_id, "reasons": ["cooldown"]}
                )
            return True
        if not self.risk.can_take_arb(self.config.max_arb_attempt_loss_per_day):
            if self.metrics_logger:
                self.metrics_logger.log_event(
                    "arb_skip",
                    payload={
                        "event_id": event_id,
                        "reasons": ["attempt_loss_limit"],
                        "attempt_loss": self.risk.arb_tracker.attempt_loss,
                    },
                )
            return True
        return False

    def evaluate_event_group(self, markets: Sequence[Dict]) -> Optional[Dict]:
        if not self.config.arb_enabled:
            return None

        if not markets:
            return None

        event_id = markets[0].get("eventId") or markets[0].get("event_id") or markets[0].get("id")
        if not event_id or self._should_skip_group(event_id):
            return None

        if self.config.arb_prefer_neg_risk_only and not any(
            NegRiskArbitrageur.is_negrisk_event(m) for m in markets
        ):
            return None

        if not self._within_batch_limit():
            if self.metrics_logger:
                self.metrics_logger.log_event(
                    "arb_skip", payload={"event_id": event_id, "reasons": ["batch_limit"]}
                )
            return None

        legs = self._collect_tokens_for_group(markets)
        if len(legs) < 3 or len(legs) > self.config.arb_max_outcomes:
            return None

        if len(legs) > self.config.max_legs_per_event:
            if self.metrics_logger:
                self.metrics_logger.log_event(
                    "arb_skip",
                    payload={"event_id": event_id, "reasons": ["too_many_legs"], "n_legs": len(legs)},
                )
            return None

        costs: List[Tuple[str, float, float, float]] = []  # token_id, price, size, robust_vwap
        depth_at_best_usdc: List[float] = []
        depth_within_band_usdc: List[float] = []
        robust_depth_sizes: List[float] = []
        depth_robustness_pass = True
        for token_id, market in legs:
            price, size, robust_size, robust_notional, robust_vwap = self._robust_depth(token_id)
            if price is None or np.isnan(price):
                return None
            size = size if size is not None else np.nan
            costs.append((token_id, price, size, robust_vwap))

            if self.config.arb_min_depth_per_leg and (np.isnan(size) or size < self.config.arb_min_depth_per_leg):
                return None

            leg_notional = price * size if size is not None and not np.isnan(size) else 0.0
            depth_at_best_usdc.append(leg_notional)
            robust_notional_usdc = robust_notional
            depth_within_band_usdc.append(robust_notional_usdc)
            robust_depth_sizes.append(robust_size)

            if self.config.min_depth_per_leg_usdc and leg_notional < self.config.min_depth_per_leg_usdc:
                if self.metrics_logger:
                    self.metrics_logger.log_event(
                        "arb_skip",
                        payload={
                            "event_id": event_id,
                            "reasons": ["insufficient_depth"],
                            "token_id": token_id,
                            "depth_usdc": leg_notional,
                        },
                    )
                return None

            if (
                self.config.depth_robustness_enabled
                and self.config.min_depth_within_band_usdc
                and robust_notional_usdc < self.config.min_depth_within_band_usdc
            ):
                if self.metrics_logger:
                    self.metrics_logger.log_event(
                        "arb_skip",
                        payload={
                            "event_id": event_id,
                            "reasons": ["insufficient_depth_within_band"],
                            "token_id": token_id,
                            "depth_within_band_usdc": robust_notional_usdc,
                        },
                    )
                return None

            if self.filter_config:
                market_state = {
                    "mid": price,
                    "spread": np.nan,
                    "depth": size,
                    "volume24h": market.get("volume24hr") or market.get("volume"),
                    "time_to_resolve": NegRiskArbitrageur._time_to_resolve(market),
                }
                ok, reasons = should_trade_market(market_state, market, now_ts=time.time(), config=self.filter_config)
                if not ok:
                    if self.metrics_logger:
                        self.metrics_logger.log_event(
                            "arb_skip",
                            payload={"event_id": event_id, "reasons": reasons, "market_id": market.get("id")},
                        )
                    return None

        buffer = self.config.edge_buffer()
        def _leg_price_wc(price: float, robust_vwap: float) -> float:
            base = robust_vwap if self.config.depth_robustness_enabled and not np.isnan(robust_vwap) else price
            return base * (1 + buffer)

        if self.config.depth_robustness_enabled:
            depth_robustness_pass = bool(depth_within_band_usdc) and (
                min(depth_within_band_usdc) >= self.config.min_depth_within_band_usdc
            )
        else:
            depth_robustness_pass = False

        total_cost = sum(price for _, price, _, _ in costs)
        total_cost_wc = sum(_leg_price_wc(price, robust_vwap) for _, price, _, robust_vwap in costs)
        edge = 1.0 - total_cost_wc
        if edge < self.config.min_edge_abs:
            return None

        max_depth = min([c[2] for c in costs if not np.isnan(c[2])], default=0.0)
        robust_depth = min(robust_depth_sizes) if robust_depth_sizes else 0.0
        base_depth = robust_depth if self.config.depth_robustness_enabled else max_depth
        if np.isnan(base_depth) or base_depth <= 0:
            return None

        size_from_event_cap = (
            self.config.max_usdc_per_event / total_cost_wc if self.config.max_usdc_per_event else base_depth
        )
        available_cycle_usdc = (
            self.config.max_usdc_per_cycle - self._cycle_notional if self.config.max_usdc_per_cycle else None
        )
        size_from_cycle_cap = (
            (available_cycle_usdc / total_cost_wc) if available_cycle_usdc is not None else base_depth
        )
        size = min(base_depth, size_from_event_cap, size_from_cycle_cap)

        if size <= 0:
            if self.metrics_logger:
                self.metrics_logger.log_event(
                    "arb_skip", payload={"event_id": event_id, "reasons": ["size_limit"], "edge": edge}
                )
            return None

        notional = total_cost_wc * size

        if not self.risk.can_trade(str(event_id), str(event_id), notional, time.time()):
            if self.metrics_logger:
                self.metrics_logger.log_event(
                    "arb_skip",
                    payload={"event_id": event_id, "reasons": ["risk_cap"], "notional": notional},
                )
            return None

        if self.metrics_logger:
            self.metrics_logger.log_event(
                "arb_opportunity",
                payload={
                    "event_id": event_id,
                    "n_outcomes": len(costs),
                    "C_wc": total_cost_wc,
                    "E_wc": edge,
                    "size_usdc": notional,
                    "depth_at_best_ask_min": min(depth_at_best_usdc) if depth_at_best_usdc else None,
                    "depth_within_band_min": min(depth_within_band_usdc) if depth_within_band_usdc else None,
                    "depth_robustness_pass": depth_robustness_pass,
                },
            )

        tif = "FOK" if self.config.use_fok_or_ioc else "IOC"
        orders = [
            (token_id, price * (1 + buffer), size) for token_id, price, size, _ in costs[: self.config.max_legs_per_event]
        ]
        fills = self.executor.place_batch_orders(orders, time_in_force=tif)

        fully_filled = all(f.get("filled_size", f.get("size", 0)) >= f.get("size", 0) for f in fills)
        if not fully_filled:
            self.executor.unwind(fills)
            filled_notional = sum(f.get("filled_size", 0) * f.get("price", 0) for f in fills)
            self.risk.record_arb_attempt_loss(filled_notional)
            self.cooldowns[event_id] = time.time() + self.config.cooldown_after_partial_fill_sec
            if self.metrics_logger:
                self.metrics_logger.log_event(
                    "arb_partial_fill",
                    payload={
                        "event_id": event_id,
                        "filled_legs": fills,
                        "attempt_loss": filled_notional,
                        "cooldown_until": self.cooldowns[event_id],
                    },
                )
            return None

        for fill in fills:
            signed_notional = fill.get("price", 0) * fill.get("filled_size", 0)
            self.risk.record_trade(str(event_id), str(event_id), "BUY_YES", signed_notional)

        self._cycle_notional += notional

        if self.metrics_logger:
            self.metrics_logger.log_event(
                "arb_trade",
                payload={"event_id": event_id, "legs": fills, "edge": edge, "size": size, "notional": notional},
            )

        return {
            "event_id": event_id,
            "edge": edge,
            "cost": total_cost,
            "size": size,
            "status": "filled",
            "legs": fills,
        }

