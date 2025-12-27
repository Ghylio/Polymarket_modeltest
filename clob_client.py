"""CLOB client aligned with Polymarket py-clob-client auth and payloads.

This module centralizes authentication (L1/L2), payload construction, and
error handling for CLOB requests while keeping public market data
endpoints unauthenticated.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional

import requests


logger = logging.getLogger(__name__)


class ClobRequestError(RuntimeError):
    """Raised when the CLOB API responds with an error."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ClobAuth:
    """Authentication material for Polymarket CLOB.

    L1 credentials sign order payloads, while L2 credentials sign HTTP
    requests that require authenticated transport.
    """

    l1_private_key: Optional[str]
    l2_api_key: Optional[str]
    l2_api_secret: Optional[str]

    @classmethod
    def from_env(cls, prefix: str = "POLYMARKET") -> "ClobAuth":
        """Load credentials from environment variables.

        Expected variables:
        - ``{prefix}_L1_PRIVATE_KEY``
        - ``{prefix}_L2_API_KEY``
        - ``{prefix}_L2_API_SECRET``
        """

        l1 = os.getenv(f"{prefix}_L1_PRIVATE_KEY")
        l2_key = os.getenv(f"{prefix}_L2_API_KEY")
        l2_secret = os.getenv(f"{prefix}_L2_API_SECRET")

        detected = [
            name
            for name, value in (
                (f"{prefix}_L1_PRIVATE_KEY", l1),
                (f"{prefix}_L2_API_KEY", l2_key),
                (f"{prefix}_L2_API_SECRET", l2_secret),
            )
            if value
        ]
        logger.info("CLOB auth env keys detected (prefix=%s): %s", prefix, detected or "none")

        return cls(l1, l2_key, l2_secret)

    def require_l1(self) -> str:
        if not self.l1_private_key:
            raise ClobRequestError("Missing L1 private key for signing orders")
        return self.l1_private_key

    def require_l2(self) -> tuple[str, str]:
        if not self.l2_api_key or not self.l2_api_secret:
            raise ClobRequestError("Missing L2 API credentials for authenticated requests")
        return self.l2_api_key, self.l2_api_secret

    def sign_l2_request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None,
                        timestamp_ms: Optional[int] = None) -> Dict[str, str]:
        """Create L2 headers following the Polymarket docs pattern."""

        api_key, api_secret = self.require_l2()
        ts = str(timestamp_ms or int(time.time() * 1000))
        serialized_body = json.dumps(body or {}, separators=(",", ":"))
        message = f"{ts}{method.upper()}{path}{serialized_body}"
        signature = hmac.new(api_secret.encode(), message.encode(), hashlib.sha256).hexdigest()
        return {
            "X-API-KEY": api_key,
            "X-SIGNATURE": signature,
            "X-TIMESTAMP": ts,
        }

    def sign_l1_payload(self, payload: Dict[str, Any]) -> str:
        """Create an order signature using the L1 private key."""

        private_key = self.require_l1()
        message = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hmac.new(private_key.encode(), message.encode(), hashlib.sha256).hexdigest()


class ClobClient:
    """Thin CLOB wrapper with standardized auth, payloads, and rounding."""

    BASE_URL = "https://clob.polymarket.com"
    TICK_SIZE = Decimal("0.0001")

    def __init__(self, auth: Optional[ClobAuth] = None, session: Optional[requests.Session] = None,
                 timeout: int = 30):
        self.auth = auth or ClobAuth.from_env()
        self.session = session or requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketModelTest/1.0",
        })
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _round_price(self, price: float) -> float:
        ticks = (Decimal(str(price)) / self.TICK_SIZE).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        rounded = (ticks * self.TICK_SIZE).quantize(self.TICK_SIZE, rounding=ROUND_HALF_UP)
        return float(rounded)

    def _request(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None,
                 json_body: Optional[Dict[str, Any]] = None, require_auth: bool = False) -> Dict[str, Any]:
        url = f"{self.BASE_URL}{path}"
        headers: Dict[str, str] = {}
        if require_auth:
            headers.update(self.auth.sign_l2_request(method, path, json_body))

        response = self.session.request(
            method,
            url,
            params=params,
            json=json_body,
            headers=headers,
            timeout=self.timeout,
        )

        if not response.ok:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise ClobRequestError(f"CLOB {method} {path} failed: {detail}", status_code=response.status_code)

        try:
            return response.json()
        except ValueError as exc:
            raise ClobRequestError("Invalid JSON in CLOB response") from exc

    # ------------------------------------------------------------------
    # Public market data (no auth)
    # ------------------------------------------------------------------
    def get_orderbook(self, token_id: str, depth: Optional[int] = None) -> Dict[str, Any]:
        params = {"token_id": token_id}
        if depth:
            params["depth"] = depth
        return self._request("GET", "/book", params=params)

    def get_price(self, token_id: str, side: str = "BUY") -> Dict[str, Any]:
        params = {"token_id": token_id, "side": side}
        return self._request("GET", "/price", params=params)

    def get_trades(self, token_id: Optional[str] = None, limit: int = 500) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if token_id:
            params["asset_id"] = token_id
        return self._request("GET", "/trades", params=params)

    # ------------------------------------------------------------------
    # Authenticated order placement
    # ------------------------------------------------------------------
    def build_order_payload(
        self,
        token_id: str,
        side: str,
        size: float,
        price: float,
        *,
        client_order_id: Optional[str] = None,
        time_in_force: str = "GTC",
        order_type: str = "LIMIT",
        expiration_seconds: int = 300,
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        rounded_price = self._round_price(price)
        now = current_time or time.time()
        payload = {
            "asset_id": token_id,
            "side": side.upper(),
            "size": float(size),
            "price": rounded_price,
            "type": order_type.upper(),
            "time_in_force": time_in_force.upper(),
            "client_order_id": client_order_id or f"modeltest-{int(now * 1000)}",
            "expiration": int(now) + expiration_seconds,
        }
        return payload

    def place_order(self, token_id: str, side: str, size: float, price: float,
                    *, client_order_id: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        base_payload = self.build_order_payload(
            token_id, side, size, price, client_order_id=client_order_id, **kwargs
        )
        signature = self.auth.sign_l1_payload(base_payload)
        payload = {**base_payload, "signature": signature}
        return self._request("POST", "/order", json_body=payload, require_auth=True)
