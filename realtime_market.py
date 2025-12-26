"""Realtime market data consumer using Polymarket websocket feed."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import websocket


@dataclass
class BookUpdate:
    asset_id: str
    bids: List[List[float]]
    asks: List[List[float]]
    last_trade_price: Optional[float] = None
    last_trade_size: Optional[float] = None
    last_trade_ts: Optional[float] = None

    @property
    def best_bid(self) -> float:
        return float(self.bids[0][0]) if self.bids else float("nan")

    @property
    def best_ask(self) -> float:
        return float(self.asks[0][0]) if self.asks else float("nan")

    @property
    def depth(self) -> float:
        return sum(float(b[1]) for b in self.bids[:5]) + sum(float(a[1]) for a in self.asks[:5])


class RealtimeMarketStream:
    """Simple websocket consumer inspired by polymarket/real-time-data-client."""

    WS_URL = "wss://clob.polymarket.com/ws"

    def __init__(
        self,
        asset_ids: Sequence[str],
        *,
        on_book: Callable[[BookUpdate], None],
        on_trade: Optional[Callable[[BookUpdate], None]] = None,
        ws_factory: Optional[Callable[..., websocket.WebSocketApp]] = None,
        reconnect_delay: float = 1.0,
    ) -> None:
        self.asset_ids = list(asset_ids)
        self.on_book = on_book
        self.on_trade = on_trade or (lambda update: None)
        self.ws_factory = ws_factory or websocket.WebSocketApp
        self.reconnect_delay = reconnect_delay

        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.connected = False
        self.last_message_ts: Optional[float] = None
        self._connection_attempts = 0

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        self._connect()

    def stop(self) -> None:
        self._stop.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    def is_healthy(self, max_age: float = 15.0) -> bool:
        if self._stop.is_set() or not self.connected:
            return False
        if self.last_message_ts is None:
            return False
        return (time.time() - self.last_message_ts) < max_age

    def update_assets(self, asset_ids: Iterable[str]) -> None:
        self.asset_ids = list(asset_ids)
        if self.connected and self._ws:
            self._send_subscribe()

    # ------------------------------------------------------------------
    def ingest_message(self, payload: Dict) -> None:
        update = BookUpdate(
            asset_id=payload.get("asset_id") or payload.get("token_id"),
            bids=payload.get("bids", []) or [],
            asks=payload.get("asks", []) or [],
            last_trade_price=payload.get("price"),
            last_trade_size=payload.get("size"),
            last_trade_ts=payload.get("timestamp"),
        )
        self.last_message_ts = time.time()
        if payload.get("type") == "trade":
            self.on_trade(update)
        else:
            self.on_book(update)

    # ------------------------------------------------------------------
    def _connect(self) -> None:
        if self._stop.is_set():
            return
        self._connection_attempts += 1
        self._ws = self.ws_factory(
            self.WS_URL,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )
        self._thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._thread.start()

    def _schedule_reconnect(self) -> None:
        if self._stop.is_set():
            return
        timer = threading.Timer(self.reconnect_delay, self._connect)
        timer.daemon = True
        timer.start()

    # ------------------------------------------------------------------
    def _send_subscribe(self) -> None:
        if not self._ws or not self.asset_ids:
            return
        try:
            msg = json.dumps({"type": "subscribe", "asset_ids": self.asset_ids})
            self._ws.send(msg)
        except Exception:
            pass

    # Websocket callbacks ------------------------------------------------
    def _on_open(self, ws):  # pragma: no cover - executed in thread
        self.connected = True
        self._send_subscribe()

    def _on_close(self, ws, *args):  # pragma: no cover - executed in thread
        self.connected = False
        if not self._stop.is_set():
            self._schedule_reconnect()

    def _on_error(self, ws, error):  # pragma: no cover - executed in thread
        self.connected = False
        if not self._stop.is_set():
            self._schedule_reconnect()

    def _on_message(self, ws, message):  # pragma: no cover - executed in thread
        try:
            payload = json.loads(message)
        except Exception:
            return
        self.ingest_message(payload)

    # Testing helpers ----------------------------------------------------
    @property
    def connection_attempts(self) -> int:
        return self._connection_attempts

