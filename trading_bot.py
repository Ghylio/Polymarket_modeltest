"""Probability-driven trading bot with risk controls and paper trading."""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from polymarket_fetcher import PolymarketFetcher
from prediction_model import PolymarketPredictor
from sentiment.store import DocumentStore
from bot.market_filters import MarketFilterConfig, should_trade_market
from metrics import MetricsLogger, create_run_dir


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TradingBotConfig:
    max_evals_per_market_per_sec: float = 1.0
    global_max_evals_per_sec: float = 50.0
    min_ticks_change_to_recalc: float = 1.0
    api_backoff_base_ms: int = 250
    api_backoff_max_ms: int = 10000
    stale_book_timeout_sec: int = 30
    stale_trade_timeout_sec: int = 120
    # Market filters
    min_24h_volume: float = 0.0
    max_spread_abs: float = 0.1
    max_spread_pct: float = 0.25
    min_top_of_book_depth: float = 0.0
    min_trades_last_24h: int = 0
    skip_if_rules_ambiguous: bool = True
    rules_min_length: int = 40
    ambiguous_keywords: Tuple[str, ...] = ("subjective", "opinion", "criteria", "discretion")
    skip_if_time_to_resolve_lt: float = 600.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def compute_backoff_ms(previous_ms: int, base_ms: int, max_ms: int) -> int:
    """Compute exponential backoff with jitter, capped at max_ms."""

    if previous_ms <= 0:
        candidate = base_ms
    else:
        candidate = min(max_ms, previous_ms * 2)
    jitter = random.randint(0, candidate)
    return min(max_ms, candidate + jitter // 2)


class TokenBucketLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, rate_per_sec: float, capacity: Optional[float] = None):
        self.rate = rate_per_sec
        self.capacity = capacity or rate_per_sec
        self.tokens = self.capacity
        self.last = time.monotonic()

    def consume(self, tokens: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self.last
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_latest_predictor(models_dir: str = "models") -> PolymarketPredictor:
    """Load the newest predictor artifact or return a fresh instance."""

    path = Path(models_dir)
    if not path.exists():
        return PolymarketPredictor()

    artifacts = sorted(path.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not artifacts:
        return PolymarketPredictor()

    return PolymarketPredictor.load_artifact(str(artifacts[0]))


# ---------------------------------------------------------------------------
# Market state handling
# ---------------------------------------------------------------------------


@dataclass
class LiveMarketState:
    """Maintain live order book derived state for a market."""

    bid: float = np.nan
    ask: float = np.nan
    mid: float = 0.5
    spread: float = 0.0
    depth: float = 0.0
    last_update: float = field(default_factory=time.time)

    @classmethod
    def from_orderbook(cls, orderbook: Dict) -> "LiveMarketState":
        bids = orderbook.get("bids", []) or []
        asks = orderbook.get("asks", []) or []

        best_bid = float(bids[0][0]) if bids else np.nan
        best_ask = float(asks[0][0]) if asks else np.nan
        depth = sum(float(b[1]) for b in bids[:5]) + sum(float(a[1]) for a in asks[:5])

        if not np.isnan(best_bid) and not np.isnan(best_ask) and best_bid > 0 and best_ask > 0:
            mid = (best_bid + best_ask) / 2
            spread = abs(best_ask - best_bid)
        else:
            mid = 0.5
            spread = 0.0

        return cls(bid=best_bid, ask=best_ask, mid=mid, spread=spread, depth=depth, last_update=time.time())


# ---------------------------------------------------------------------------
# Risk management and paper trading
# ---------------------------------------------------------------------------


@dataclass
class RiskManager:
    per_market_cap: float = 500.0
    per_event_cap: float = 1000.0
    daily_loss_limit: float = 1000.0
    stale_seconds: float = 300.0
    cumulative_pnl: float = 0.0
    market_exposure: Dict[str, float] = field(default_factory=dict)
    event_exposure: Dict[str, float] = field(default_factory=dict)

    def can_trade(self, market_id: str, event_id: str, notional: float, last_update: float) -> bool:
        if time.time() - last_update > self.stale_seconds:
            return False

        m_exp = abs(self.market_exposure.get(market_id, 0.0))
        e_exp = abs(self.event_exposure.get(event_id, 0.0))

        if m_exp + notional > self.per_market_cap:
            return False
        if e_exp + notional > self.per_event_cap:
            return False
        if self.cumulative_pnl <= -self.daily_loss_limit:
            return False
        return True

    def record_trade(self, market_id: str, event_id: str, side: str, notional: float):
        signed = notional if side in {"BUY_YES", "SELL_NO"} else -notional
        self.market_exposure[market_id] = self.market_exposure.get(market_id, 0.0) + signed
        self.event_exposure[event_id] = self.event_exposure.get(event_id, 0.0) + signed

    def mark_pnl(self, delta: float):
        self.cumulative_pnl += delta

    def cancel_all(self):
        self.market_exposure.clear()
        self.event_exposure.clear()


@dataclass
class PaperBroker:
    """Simple paper-trading ledger."""

    cash: float = 10000.0
    positions: Dict[str, float] = field(default_factory=dict)

    def execute(self, market_id: str, side: str, price: float, size: float) -> float:
        notional = price * size
        signed_size = size if side in {"BUY_YES", "SELL_NO"} else -size
        self.positions[market_id] = self.positions.get(market_id, 0.0) + signed_size
        self.cash -= notional if signed_size > 0 else -notional
        return notional


# ---------------------------------------------------------------------------
# Trading bot
# ---------------------------------------------------------------------------


class ProbabilityTradingBot:
    def __init__(
        self,
        fetcher: Optional[PolymarketFetcher] = None,
        models_dir: str = None,
        threshold: float = 0.05,
        paper_trading: bool = True,
        risk_config: Optional[Dict] = None,
        bot_config: Optional[TradingBotConfig] = None,
        sentiment_db_path: Optional[Path | str] = None,
        sentiment_store: Optional[DocumentStore] = None,
        run_dir: Optional[str] = None,
        metrics_logger: Optional[MetricsLogger] = None,
    ):
        self.fetcher = fetcher or PolymarketFetcher(verbose=False)
        models_dir = models_dir or os.environ.get("POLYMARKET_MODELS", "models")
        self.run_dir = Path(run_dir) if run_dir else create_run_dir()
        self.metrics_logger = metrics_logger or MetricsLogger(self.run_dir)
        self.predictor = load_latest_predictor(models_dir)
        self.predictor.metrics_logger = self.metrics_logger
        self.sentiment_store = sentiment_store or self._maybe_init_store(sentiment_db_path)
        self.predictor.set_sentiment_store(self.sentiment_store)
        self.threshold = threshold
        self.paper = paper_trading
        self.risk = RiskManager(**(risk_config or {}))
        self.broker = PaperBroker() if paper_trading else None
        self.live_state: Dict[str, LiveMarketState] = {}
        self.bot_config = bot_config or TradingBotConfig()
        self.filter_config = MarketFilterConfig(
            min_24h_volume=self.bot_config.min_24h_volume,
            max_spread_abs=self.bot_config.max_spread_abs,
            max_spread_pct=self.bot_config.max_spread_pct,
            min_top_of_book_depth=self.bot_config.min_top_of_book_depth,
            min_trades_last_24h=self.bot_config.min_trades_last_24h,
            skip_if_rules_ambiguous=self.bot_config.skip_if_rules_ambiguous,
            rules_min_length=self.bot_config.rules_min_length,
            ambiguous_keywords=self.bot_config.ambiguous_keywords,
            skip_if_time_to_resolve_lt=self.bot_config.skip_if_time_to_resolve_lt,
        )
        self.last_eval: Dict[str, Dict] = {}
        self.backoff_until: Dict[str, float] = {}
        self.backoff_ms: Dict[str, int] = {}
        self.global_limiter = TokenBucketLimiter(
            rate_per_sec=self.bot_config.global_max_evals_per_sec,
            capacity=self.bot_config.global_max_evals_per_sec,
        )
        self.skip_reasons: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def _maybe_init_store(self, sentiment_db_path: Optional[Path | str]) -> Optional[DocumentStore]:
        path = Path(sentiment_db_path) if sentiment_db_path else Path(os.environ.get("SENTIMENT_DB", "data/sentiment.db"))
        if not path.exists():
            logger.info("Sentiment store %s not found; proceeding with NaN sentiment", path)
            return None
        try:
            return DocumentStore(path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to initialize sentiment store %s: %s", path, exc)
            return None

    # ------------------------------------------------------------------
    def _get_market_state(self, market: Dict) -> Tuple[Optional[str], LiveMarketState]:
        yes_token, _ = self.fetcher.get_token_ids_for_market(market)
        market_id = market.get("id") or yes_token
        if not yes_token:
            return None, LiveMarketState()

        cached = self.live_state.get(market_id)
        if cached:
            # Kill switch if feed is stale
            if time.time() - cached.last_update > self.bot_config.stale_book_timeout_sec:
                return yes_token, cached
            return yes_token, cached

        book = self.fetcher.get_orderbook(yes_token)
        state = LiveMarketState.from_orderbook(book)
        self.live_state[market_id] = state
        return yes_token, state

    def _get_trades_df(self, token_id: str, mid: float) -> pd.DataFrame:
        trades = self.fetcher.get_trades(token_id, limit=200)
        if trades:
            df = self.fetcher.trades_to_dataframe(trades)
            return df
        return pd.DataFrame({"price": [mid], "size": [0.0], "timestamp": [pd.Timestamp.utcnow()]})

    @staticmethod
    def _recent_trades_count(trades_df: pd.DataFrame, now_ts: Optional[float] = None) -> Optional[int]:
        if trades_df is None or trades_df.empty:
            return 0
        now_ts = now_ts or time.time()
        try:
            ts = pd.to_datetime(trades_df["timestamp"]).astype("int64") / 1e9
            return int((ts >= now_ts - 24 * 3600).sum())
        except Exception:
            return None

    @staticmethod
    def _time_to_resolve(market: Dict, now_ts: Optional[float] = None) -> Optional[float]:
        now_ts = now_ts or time.time()
        keys = ["endTime", "endDate", "resolutionTime", "closeTime"]
        for key in keys:
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

    def _is_data_stale(self, state: LiveMarketState, trades_df: pd.DataFrame, market_id: Optional[str] = None) -> bool:
        now = time.time()
        if now - state.last_update > self.bot_config.stale_book_timeout_sec:
            logger.warning("Order book stale for market; pausing decisions")
            if self.metrics_logger:
                self.metrics_logger.log_event(
                    "health",
                    payload={"reason": "stale_book", "market_id": market_id},
                )
            return True

        if self.bot_config.stale_trade_timeout_sec:
            try:
                latest_trade_ts = pd.to_datetime(trades_df["timestamp"].max()).timestamp()
                if now - latest_trade_ts > self.bot_config.stale_trade_timeout_sec:
                    logger.warning("Trades stale for market; pausing decisions")
                    if self.metrics_logger:
                        self.metrics_logger.log_event(
                            "health",
                            payload={"reason": "stale_trades"},
                        )
                    return True
            except Exception:
                # If timestamps malformed, err on the side of allowing evaluation
                return False
        return False

    def _should_evaluate(self, market_id: str, state: LiveMarketState) -> bool:
        last = self.last_eval.get(market_id)
        now = time.time()
        if not last:
            return True

        bid_move = abs(state.bid - last.get("bid", np.nan)) if not np.isnan(state.bid) and not np.isnan(last.get("bid", np.nan)) else np.inf
        ask_move = abs(state.ask - last.get("ask", np.nan)) if not np.isnan(state.ask) and not np.isnan(last.get("ask", np.nan)) else np.inf
        tick_move = max(bid_move, ask_move)

        if tick_move >= self.bot_config.min_ticks_change_to_recalc:
            return True

        min_interval = 1.0 / self.bot_config.max_evals_per_market_per_sec if self.bot_config.max_evals_per_market_per_sec > 0 else 0
        if now - last.get("time", 0) >= min_interval:
            return True

        logger.info("Skip eval %s due to per-market throttle", market_id)
        if self.metrics_logger:
            self.metrics_logger.log_event(
                "health",
                payload={"market_id": market_id, "reason": "per_market_throttle"},
            )
        return False

    def predict_market_prob(
        self, market: Dict, token_id: str, state: LiveMarketState, trades_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        trades_df = trades_df if trades_df is not None else self._get_trades_df(token_id, state.mid)
        prob = self.predictor.predict_probability(market, trades_df)
        prob.update({
            "bid": state.bid,
            "ask": state.ask,
            "mid": state.mid,
            "spread": state.spread,
            "depth": state.depth,
        })
        return prob

    def _trade_decision(self, prob: Dict, state: LiveMarketState) -> Optional[Dict]:
        bid, ask = prob.get("bid"), prob.get("ask")
        p_lcb, p_ucb = prob.get("p_lcb", 0.5), prob.get("p_ucb", 0.5)

        if not np.isnan(ask) and (p_lcb - ask) > self.threshold:
            return {"side": "BUY_YES", "price": ask}
        if not np.isnan(bid) and (bid - p_ucb) > self.threshold:
            return {"side": "SELL_YES", "price": bid}
        return None

    def process_market(self, market: Dict) -> Optional[Dict]:
        token_id, state = self._get_market_state(market)
        if not token_id:
            return None

        market_id = market.get("id", token_id)

        # Backoff handling
        if self.backoff_until.get(market_id, 0) > time.time():
            logger.info("Skipping %s due to backoff until %.2f", market_id, self.backoff_until[market_id])
            if self.metrics_logger:
                self.metrics_logger.log_event(
                    "health",
                    payload={"market_id": market_id, "reason": "backoff_active", "backoff_until": self.backoff_until[market_id]},
                )
            return None

        if not self._should_evaluate(market_id, state):
            return None

        if not self.global_limiter.consume():
            logger.info("Global eval limiter reached; skipping evaluation")
            if self.metrics_logger:
                self.metrics_logger.log_event(
                    "health",
                    payload={"reason": "global_throttle", "market_id": market_id},
                )
            return None

        try:
            trades_df = self._get_trades_df(token_id, state.mid)
            market_state = {
                "bid": state.bid,
                "ask": state.ask,
                "mid": state.mid,
                "spread": state.spread,
                "depth": state.depth,
                "volume24h": market.get("volume24hr") or market.get("volume"),
                "trades_last_24h": self._recent_trades_count(trades_df),
                "time_to_resolve": self._time_to_resolve(market),
            }
            ok, reasons = should_trade_market(
                market_state, market, now_ts=time.time(), config=self.filter_config
            )
            if not ok:
                for reason in reasons:
                    self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1
                if self.metrics_logger:
                    self.metrics_logger.log_event(
                        "filter_skip",
                        payload={"market_id": market_id, "reasons": reasons},
                    )
                logger.info("Skipping market %s due to filters: %s", market_id, ",".join(reasons))
                return None
            if self._is_data_stale(state, trades_df, market_id=market_id):
                if not self.paper:
                    self.risk.cancel_all()
                logger.warning("Pausing market %s due to stale data", market_id)
                return None
            prob = self.predict_market_prob(market, token_id, state, trades_df=trades_df)
        except Exception:
            delay = compute_backoff_ms(
                self.backoff_ms.get(market_id, 0),
                self.bot_config.api_backoff_base_ms,
                self.bot_config.api_backoff_max_ms,
            )
            self.backoff_ms[market_id] = delay
            self.backoff_until[market_id] = time.time() + delay / 1000.0
            logger.exception("Error during probability evaluation; backing off %s ms", delay)
            if self.metrics_logger:
                self.metrics_logger.log_event(
                    "health",
                    payload={"market_id": market_id, "reason": "backoff_error", "backoff_ms": delay},
                )
            return None

        decision = self._trade_decision(prob, state)
        if not decision:
            return None

        event_id = market.get("eventId") or market_id
        notional = decision["price"] * 1.0  # unit size placeholder

        if not self.risk.can_trade(market_id, event_id, notional, state.last_update):
            if self.metrics_logger:
                self.metrics_logger.log_event(
                    "decision",
                    payload={
                        "market_id": market_id,
                        "action": "blocked_risk",
                        "threshold": self.threshold,
                        "reason": "risk_cap",
                    },
                )
            return None

        if self.metrics_logger:
            edge = prob.get("p_lcb") - decision["price"] if decision["side"] == "BUY_YES" else decision["price"] - prob.get("p_ucb")
            self.metrics_logger.log_event(
                "decision",
                payload={
                    "market_id": market_id,
                    "action": decision["side"],
                    "size": 1.0,
                    "threshold": self.threshold,
                    "edge": edge,
                },
            )

        if self.paper:
            executed = self.broker.execute(market_id, decision["side"], decision["price"], size=1.0)
            self.risk.record_trade(market_id, event_id, decision["side"], executed)
            pnl = 0.0
        else:
            self.risk.record_trade(market_id, event_id, decision["side"], notional)
            pnl = 0.0
        if self.metrics_logger:
            self.metrics_logger.log_event(
                "trade",
                payload={
                    "market_id": market_id,
                    "action": decision["side"],
                    "price": decision["price"],
                    "size": 1.0,
                    "pnl": pnl,
                },
            )
        self.last_eval[market_id] = {"time": time.time(), "bid": state.bid, "ask": state.ask}
        self.backoff_ms[market_id] = 0
        self.backoff_until[market_id] = 0

        return {"market_id": market_id, **prob, **decision}

    def run_once(self, limit: int = 10) -> List[Dict]:
        markets = self.fetcher.get_markets(limit=limit, order="volume24hr")
        executed: List[Dict] = []
        for market in markets:
            try:
                result = self.process_market(market)
                if result:
                    executed.append(result)
            except Exception:
                continue
        return executed

    def shutdown(self):
        self.risk.cancel_all()

