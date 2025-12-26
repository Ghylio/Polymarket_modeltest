"""Batch order executor with deterministic signing and serialization.

This mirrors the safety checks used by Polymarket's clob-order-utils by
enforcing canonical serialization, deterministic hashing/signing, and strict
batch sizing before submitting through the CLOB HTTP client.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from clob_client import ClobAuth, ClobClient
from negrisk_arb import OrderExecutor


def _format_decimal(value: float | Decimal, *, quantize_to: Decimal | None = None) -> str:
    dec = Decimal(str(value))
    if quantize_to is not None:
        dec = dec.quantize(quantize_to, rounding=ROUND_HALF_UP)
    normalized = dec.normalize()
    # Ensure trailing zeros are removed while preserving small decimals
    return format(normalized, "f") if normalized.as_tuple().exponent < 0 else str(normalized)


def _canonical_dumps(payload: Dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass
class BatchOrder:
    order: Dict
    signature: str
    order_hash: str


class ClobBatchExecutor(OrderExecutor):
    """Submit deterministic CLOB batches with canonicalized orders."""

    def __init__(
        self,
        client: ClobClient | None = None,
        *,
        max_batch_legs: int = 15,
        expiration_seconds: int = 300,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.client = client or ClobClient()
        self.auth: ClobAuth = self.client.auth
        self.max_batch_legs = max_batch_legs
        self.expiration_seconds = expiration_seconds
        self.clock = clock

    # ------------------------------------------------------------------
    # Canonicalization helpers
    # ------------------------------------------------------------------
    def _canonical_order(self, payload: Dict) -> Dict:
        canonical: Dict = {}
        for key in sorted(payload.keys()):
            value = payload[key]
            if key == "price":
                canonical[key] = _format_decimal(value, quantize_to=self.client.TICK_SIZE)
            elif key == "size":
                canonical[key] = _format_decimal(value)
            else:
                canonical[key] = value
        return canonical

    def _hash_order(self, order: Dict) -> str:
        return hashlib.sha256(_canonical_dumps(order).encode()).hexdigest()

    def _sign_order(self, order: Dict) -> str:
        return self.auth.sign_l1_payload(order)

    # ------------------------------------------------------------------
    # Batch construction and submission
    # ------------------------------------------------------------------
    def build_batch_payload(
        self, orders: Sequence[Tuple[str, float, float]], *, time_in_force: str = "FOK"
    ) -> Dict[str, List[Dict]]:
        if len(orders) > self.max_batch_legs:
            raise ValueError(f"Batch size {len(orders)} exceeds max legs {self.max_batch_legs}")

        batch: List[Dict] = []
        now = self.clock()
        for token_id, price, size in orders:
            base = self.client.build_order_payload(
                token_id,
                side="BUY",
                size=size,
                price=price,
                time_in_force=time_in_force,
                order_type="LIMIT",
                expiration_seconds=self.expiration_seconds,
                current_time=now,
            )
            canonical = self._canonical_order(base)
            order_hash = self._hash_order(canonical)
            signature = self._sign_order(canonical)
            batch.append({"order": canonical, "orderHash": order_hash, "signature": signature})

        return {"orders": batch}

    def place_batch_orders(
        self, orders: Sequence[Tuple[str, float, float]], time_in_force: str = "FOK"
    ) -> List[Dict]:
        payload = self.build_batch_payload(orders, time_in_force=time_in_force)
        response = self.client._request(
            "POST", "/orders/batch", json_body=payload, require_auth=True
        )

        # Prefer returning the order objects so the strategy can track status/fills.
        if isinstance(response, dict) and isinstance(response.get("orders"), Iterable):
            return list(response.get("orders") or [])
        if isinstance(response, list):
            return response
        return [response]

    def unwind(self, fills: Sequence[Dict]):
        # Best-effort: submit cancellation for each filled leg.
        # For now this is a no-op; cancellation wiring would rely on a cancel endpoint.
        return []
