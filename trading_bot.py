"""Probability-driven trading bot with risk controls and paper trading."""

from __future__ import annotations

import logging
import os
import queue
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml

from polymarket_fetcher import PolymarketFetcher
from prediction_model import PolymarketPredictor
from realtime_market import BookUpdate, RealtimeMarketStream
from sentiment.store import DocumentStore
from research.store import ResearchFeatureStore
from bot.market_filters import MarketFilterConfig, should_trade_market
from bot.strategy_structural_arb import StructuralArbConfig, StructuralArbStrategy
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
    # Structural arbitrage
    structural_arb: StructuralArbConfig = field(default_factory=StructuralArbConfig)


def load_trading_bot_config(path: Path | str = "config/trading_bot_config.yaml") -> TradingBotConfig:
    """Load TradingBotConfig from YAML, applying structural_arb defaults."""

    path = Path(path)
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    if data is None:
        data = {}

    structural_payload = data.get("structural_arb", {}) or {}

    config_kwargs = {k: v for k, v in data.items() if k != "structural_arb"}
    cfg = TradingBotConfig(**config_kwargs)

    # Normalize ambiguous keywords to tuple for consistency
    if isinstance(cfg.ambiguous_keywords, list):
        cfg.ambiguous_keywords = tuple(cfg.ambiguous_keywords)

    cfg.structural_arb = StructuralArbConfig(**structural_payload)
    return cfg


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
    last_trade_price: Optional[float] = None
    last_trade_size: Optional[float] = None
    last_trade_ts: Optional[float] = None
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

    @classmethod
    def from_realtime(cls, update: BookUpdate) -> "LiveMarketState":
        best_bid = update.best_bid
        best_ask = update.best_ask
        if not np.isnan(best_bid) and not np.isnan(best_ask) and best_bid > 0 and best_ask > 0:
            mid = (best_bid + best_ask) / 2
            spread = abs(best_ask - best_bid)
        else:
            mid = 0.5
            spread = 0.0

        return cls(
            bid=best_bid,
            ask=best_ask,
            mid=mid,
            spread=spread,
            depth=update.depth,
            last_trade_price=update.last_trade_price,
            last_trade_size=update.last_trade_size,
            last_trade_ts=update.last_trade_ts,
            last_update=time.time(),
        )


# ---------------------------------------------------------------------------
# Risk management and paper trading
# ---------------------------------------------------------------------------


@dataclass
class ArbRiskTracker:
    attempt_loss: float = 0.0
    last_reset_day: int = field(default_factory=lambda: time.gmtime().tm_yday)

    def _reset_if_new_day(self) -> None:
        today = time.gmtime().tm_yday
        if today != self.last_reset_day:
            self.attempt_loss = 0.0
            self.last_reset_day = today

    def can_attempt(self, max_loss: float) -> bool:
        self._reset_if_new_day()
        if max_loss <= 0:
            return True
        return self.attempt_loss < max_loss

    def record_attempt_loss(self, loss: float) -> None:
        self._reset_if_new_day()
        self.attempt_loss += max(loss, 0.0)


@dataclass
class RiskManager:
    per_market_cap: float = 500.0
    per_event_cap: float = 1000.0
    daily_loss_limit: float = 1000.0
    stale_seconds: float = 300.0
    cumulative_pnl: float = 0.0
    market_exposure: Dict[str, float] = field(default_factory=dict)
    event_exposure: Dict[str, float] = field(default_factory=dict)
    arb_tracker: ArbRiskTracker = field(default_factory=ArbRiskTracker)

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

    def can_take_arb(self, max_attempt_loss_per_day: float) -> bool:
        return self.arb_tracker.can_attempt(max_attempt_loss_per_day)

    def record_arb_attempt_loss(self, loss: float) -> None:
        self.arb_tracker.record_attempt_loss(loss)

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
        research_db_path: Optional[Path | str] = None,
        research_store: Optional[ResearchFeatureStore] = None,
        use_research: bool = False,
        run_dir: Optional[str] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        enable_realtime: bool = True,
    ):
        self.fetcher = fetcher or PolymarketFetcher(verbose=False)
        models_dir = models_dir or os.environ.get("POLYMARKET_MODELS", "models")
        self.run_dir = Path(run_dir) if run_dir else create_run_dir()
        self.metrics_logger = metrics_logger or MetricsLogger(self.run_dir)
        self.predictor = load_latest_predictor(models_dir)
        self.predictor.metrics_logger = self.metrics_logger
        self.sentiment_store = sentiment_store or self._maybe_init_store(sentiment_db_path)
        self.predictor.set_sentiment_store(self.sentiment_store)
        self.research_store = research_store or (
            self._maybe_init_research_store(research_db_path) if use_research else None
        )
        if self.research_store:
            self.predictor.set_research_store(self.research_store)
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
        arb_config = self.bot_config.structural_arb
        self.arb_filter_config = MarketFilterConfig(
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
        self.arb_strategy = StructuralArbStrategy(
            fetcher=self.fetcher,
            risk_manager=self.risk,
            config=arb_config,
            filter_config=self.arb_filter_config,
            metrics_logger=self.metrics_logger,
        )
        self.last_eval: Dict[str, Dict] = {}
        self.backoff_until: Dict[str, float] = {}
        self.backoff_ms: Dict[str, int] = {}
        self.market_update_queue: "queue.Queue[str]" = queue.Queue()
        self.realtime_enabled = enable_realtime
        self.realtime_client: Optional[RealtimeMarketStream] = None
        self.market_by_id: Dict[str, Dict] = {}
        self.token_to_market_id: Dict[str, str] = {}
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

    def _maybe_init_research_store(
        self, research_db_path: Optional[Path | str]
    ) -> Optional[ResearchFeatureStore]:
        path = Path(research_db_path) if research_db_path else Path(os.environ.get("RESEARCH_DB", "data/sentiment.db"))
        if not path.exists():
            logger.info("Research store %s not found; proceeding with default research features", path)
            return None
        try:
            return ResearchFeatureStore(path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to initialize research store %s: %s", path, exc)
            return None

    # ------------------------------------------------------------------
    def _index_markets(self, markets: List[Dict]) -> None:
        for market in markets:
            market_id = market.get("id") or market.get("conditionId")
            yes_token, _ = self.fetcher.get_token_ids_for_market(market)
            if market_id and yes_token:
                self.market_by_id[str(market_id)] = market
                self.token_to_market_id[str(yes_token)] = str(market_id)

    def _ensure_realtime(self) -> None:
        if not self.realtime_enabled or not self.token_to_market_id:
            return
        if not self.realtime_client:
            self.realtime_client = RealtimeMarketStream(
                list(self.token_to_market_id.keys()),
                on_book=self._on_realtime_book,
                on_trade=self._on_realtime_trade,
            )
            self.realtime_client.start()
        else:
            self.realtime_client.update_assets(self.token_to_market_id.keys())

    def _on_realtime_book(self, update: BookUpdate) -> None:
        market_id = self.token_to_market_id.get(update.asset_id)
        if not market_id:
            return
        state = LiveMarketState.from_realtime(update)
        # Preserve previous last trade info if this update lacked trades
        previous = self.live_state.get(market_id)
        if previous and update.last_trade_ts is None:
            state.last_trade_price = previous.last_trade_price
            state.last_trade_size = previous.last_trade_size
            state.last_trade_ts = previous.last_trade_ts
        self.live_state[market_id] = state
        try:
            self.market_update_queue.put_nowait(market_id)
        except queue.Full:  # pragma: no cover - queue unbounded by default
            pass

    def _on_realtime_trade(self, update: BookUpdate) -> None:
        market_id = self.token_to_market_id.get(update.asset_id)
        if not market_id:
            return
        existing = self.live_state.get(market_id) or LiveMarketState()
        existing.last_trade_price = update.last_trade_price
        existing.last_trade_size = update.last_trade_size
        existing.last_trade_ts = update.last_trade_ts
        existing.last_update = time.time()
        self.live_state[market_id] = existing
        try:
            self.market_update_queue.put_nowait(market_id)
        except queue.Full:  # pragma: no cover
            pass

    def _drain_market_updates(self) -> Set[str]:
        updates: Set[str] = set()
        while True:
            try:
                updates.add(self.market_update_queue.get_nowait())
            except queue.Empty:
                break
        return updates

    # ------------------------------------------------------------------
    def _get_market_state(self, market: Dict) -> Tuple[Optional[str], LiveMarketState]:
        yes_token, _ = self.fetcher.get_token_ids_for_market(market)
        market_id = market.get("id") or yes_token
        if not yes_token:
            return None, LiveMarketState()

        cached = self.live_state.get(market_id)
        if cached:
            if self.realtime_client and self.realtime_client.is_healthy():
                return yes_token, cached
            if time.time() - cached.last_update <= self.bot_config.stale_book_timeout_sec:
                return yes_token, cached

        book = self.fetcher.get_orderbook(yes_token)
        state = LiveMarketState.from_orderbook(book)
        self.live_state[market_id] = state
        return yes_token, state

    def _get_trades_df(self, token_id: str, mid: float, state: Optional[LiveMarketState] = None) -> pd.DataFrame:
        if state and state.last_trade_ts:
            return pd.DataFrame(
                {
                    "price": [state.last_trade_price if state.last_trade_price is not None else mid],
                    "size": [state.last_trade_size or 0.0],
                    "timestamp": [pd.to_datetime(state.last_trade_ts, unit="s")],
                }
            )

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

    def _evaluate_structural_arbitrage(self, markets: List[Dict]) -> Set[str]:
        executed_events: Set[str] = set()
        if not self.bot_config.structural_arb.arb_enabled:
            return executed_events

        self.arb_strategy.reset_cycle()
        groups: Dict[str, List[Dict]] = {}
        for market in markets:
            event_id = market.get("eventId") or market.get("event_id") or market.get("conditionId")
            if not event_id:
                continue
            groups.setdefault(event_id, []).append(market)

        for event_id, group_markets in groups.items():
            if len(group_markets) < 2:
                continue
            result = self.arb_strategy.evaluate_event_group(group_markets)
            if result and result.get("status") == "filled":
                executed_events.add(str(event_id))

        return executed_events

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
            trades_df = self._get_trades_df(token_id, state.mid, state=state)
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
        self._index_markets(markets)
        if self.realtime_enabled:
            self._ensure_realtime()

        executed: List[Dict] = []
        arb_events = self._evaluate_structural_arbitrage(markets)
        healthy = self.realtime_client.is_healthy() if self.realtime_client else False
        target_markets: Set[str] = self._drain_market_updates() if healthy else set()
        if not healthy:
            target_markets = {str(m.get("id") or self.fetcher.get_token_ids_for_market(m)[0]) for m in markets}

        for market in markets:
            try:
                market_id = str(market.get("id") or market.get("conditionId") or "")
                event_id = market.get("eventId") or market.get("event_id") or market.get("conditionId")
                if event_id and str(event_id) in arb_events:
                    continue
                if target_markets and market_id not in target_markets:
                    continue
                result = self.process_market(market)
                if result:
                    executed.append(result)
            except Exception:
                continue
        return executed

    def shutdown(self):
        self.risk.cancel_all()
        if self.realtime_client:
            self.realtime_client.stop()

