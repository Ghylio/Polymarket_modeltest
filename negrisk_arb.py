"""NegRisk full-set arbitrage strategy module."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from polymarket_fetcher import PolymarketFetcher
from bot.market_filters import MarketFilterConfig, should_trade_market

if TYPE_CHECKING:  # pragma: no cover
    from trading_bot import RiskManager


@dataclass
class NegRiskConfig:
    buffer: float = 0.02
    size_per_outcome: float = 1.0


class OrderExecutor:
    """Abstract order executor for batch submission."""

    def place_batch_orders(
        self, orders: Sequence[Tuple[str, float, float]], time_in_force: str = "FOK"
    ) -> List[Dict]:
        raise NotImplementedError

    def unwind(self, fills: Sequence[Dict]):
        """Unwind partially filled legs (best-effort)."""
        raise NotImplementedError


class PaperBatchExecutor(OrderExecutor):
    """Paper executor that fills all orders at quoted prices."""

    def place_batch_orders(
        self, orders: Sequence[Tuple[str, float, float]], time_in_force: str = "FOK"
    ) -> List[Dict]:
        fills = []
        for token_id, price, size in orders:
            fills.append(
                {
                    "token_id": token_id,
                    "price": price,
                    "size": size,
                    "filled_size": size,
                    "status": "filled",
                    "tif": time_in_force,
                }
            )
        return fills

    def unwind(self, fills: Sequence[Dict]):
        # Nothing to unwind in paper mode; assume instant cancellation.
        return []


class NegRiskArbitrageur:
    def __init__(
        self,
        fetcher: Optional[PolymarketFetcher] = None,
        executor: Optional[OrderExecutor] = None,
        risk_manager: Optional["RiskManager"] = None,
        config: Optional[NegRiskConfig] = None,
        filter_config: Optional[MarketFilterConfig] = None,
        apply_filters: bool = True,
    ):
        self.fetcher = fetcher or PolymarketFetcher(verbose=False)
        self.executor = executor or PaperBatchExecutor()
        if risk_manager is None:
            from trading_bot import RiskManager

            self.risk = RiskManager()
        else:
            self.risk = risk_manager
        self.config = config or NegRiskConfig()
        self.filter_config = filter_config or MarketFilterConfig()
        self.apply_filters = apply_filters
        self.skip_reasons: Dict[str, int] = {}

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_tokens(market: Dict) -> List[str]:
        tokens_raw = market.get("clobTokenIds") or []
        if isinstance(tokens_raw, str):
            try:
                tokens = json.loads(tokens_raw)
            except Exception:
                tokens = []
        else:
            tokens = tokens_raw
        return [t for t in tokens if t]

    @staticmethod
    def is_negrisk_event(market: Dict) -> bool:
        markers = {
            str(market.get("type", "")) .lower(),
            str(market.get("eventType", "")) .lower(),
            str(market.get("category", "")) .lower(),
        }
        flags = {"negrisk", "neg_risk", "negative risk"}
        if any(m in flags for m in markers):
            return True
        return bool(market.get("isNegRisk") or market.get("negRisk"))

    @staticmethod
    def _time_to_resolve(market: Dict, now_ts: Optional[float] = None) -> Optional[float]:
        now_ts = now_ts or time.time()
        for key in ["endTime", "endDate", "resolutionTime", "closeTime"]:
            val = market.get(key)
            if val is None:
                continue
            try:
                ts = float(val)
                if ts > 1e12:
                    ts = ts / 1000.0
                return ts - now_ts
            except Exception:
                continue
        return None

    def _full_set_cost_with_depth(
        self, market: Dict
    ) -> Tuple[Optional[float], List[Tuple[str, float]], Optional[float]]:
        tokens = self._parse_tokens(market)
        if len(tokens) < 2:
            return None, [], None

        prices: List[Tuple[str, float]] = []
        total = 0.0
        top_depth = []
        for token in tokens:
            ob = self.fetcher.get_orderbook(token)
            asks = ob.get("asks", []) if ob else []
            best_ask = None
            size = np.nan
            if asks:
                first = asks[0]
                if isinstance(first, dict):
                    best_ask = float(first.get("price") or first.get("[0]", np.nan))
                    size = float(first.get("size") or first.get("[1]", np.nan))
                else:
                    best_ask = float(first[0])
                    size = float(first[1]) if len(first) > 1 else np.nan
            if best_ask is None or np.isnan(best_ask):
                return None, [], None
            prices.append((token, best_ask))
            total += best_ask
            if not np.isnan(size):
                top_depth.append(size)

        depth = min(top_depth) if top_depth else None
        return total, prices, depth

    def compute_full_set_cost(self, market: Dict) -> Tuple[Optional[float], List[Tuple[str, float]]]:
        total, prices, _ = self._full_set_cost_with_depth(market)
        return total, prices

    def evaluate_and_trade(self, market: Dict) -> Optional[Dict]:
        if not self.is_negrisk_event(market):
            return None

        market_id = market.get("id") or market.get("conditionId")
        event_id = market.get("eventId") or market_id

        cost, prices, depth = self._full_set_cost_with_depth(market)
        if cost is None or not prices:
            return None

        if self.apply_filters:
            market_state = {
                "mid": cost / len(prices) if prices else np.nan,
                "spread": np.nan,
                "depth": depth,
                "volume24h": market.get("volume24hr") or market.get("volume"),
                "time_to_resolve": self._time_to_resolve(market),
            }
            ok, reasons = should_trade_market(
                market_state, market, now_ts=time.time(), config=self.filter_config
            )
            if not ok:
                for reason in reasons:
                    self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1
                return None

        edge = 1.0 - cost
        if edge <= self.config.buffer:
            return None

        notional = cost * self.config.size_per_outcome
        now = time.time()
        if not self.risk.can_trade(market_id, event_id, notional, now):
            return None

        orders = [
            (token_id, price, self.config.size_per_outcome)
            for token_id, price in prices
        ]
        fills = self.executor.place_batch_orders(orders, time_in_force="FOK")

        fully_filled = all(f.get("filled_size", 0) >= f.get("size", 0) for f in fills)
        if not fully_filled:
            self.executor.unwind(fills)
            return {
                "market_id": market_id,
                "event_id": event_id,
                "cost": cost,
                "edge": edge,
                "status": "unwound",
            }

        for fill in fills:
            self.risk.record_trade(market_id, event_id, "BUY_YES", fill.get("price", 0) * fill.get("filled_size", 0))

        return {
            "market_id": market_id,
            "event_id": event_id,
            "cost": cost,
            "edge": edge,
            "status": "filled",
        }

