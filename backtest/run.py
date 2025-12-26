"""Replay backtester for probability-driven bot on snapshot datasets."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bot.market_filters import MarketFilterConfig, should_trade_market
from prediction_model import PolymarketPredictor
from trading_bot import PaperBroker, RiskManager, TradingBotConfig
from metrics import MetricsLogger

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _extract_event_id(row: pd.Series) -> Optional[str]:
    for key in ("event_id", "eventId", "group_id", "groupId"):
        if key in row and pd.notna(row[key]):
            return str(row[key])
    for key in ("market_id", "id"):
        if key in row and pd.notna(row[key]):
            return str(row[key])
    return None


def _extract_token_id(row: pd.Series) -> Optional[str]:
    for key in ("token_id", "tokenId", "clobTokenId", "clob_token_id"):
        if key in row and pd.notna(row[key]):
            return str(row[key])
    return None


def _price_from_series(row: pd.Series) -> Tuple[float, float, float]:
    bid = float(row.get("bid", np.nan))
    ask = float(row.get("ask", np.nan))
    p_mid = float(row.get("p_mid", np.nan))
    spread = float(row.get("spread", np.nan)) if not pd.isna(row.get("spread")) else np.nan

    if np.isnan(bid) and not np.isnan(p_mid) and not np.isnan(spread):
        bid = max(p_mid - spread / 2, 0.0)
    if np.isnan(ask) and not np.isnan(p_mid) and not np.isnan(spread):
        ask = min(p_mid + spread / 2, 1.0)
    if np.isnan(p_mid) and not np.isnan(bid) and not np.isnan(ask):
        p_mid = (bid + ask) / 2
    if np.isnan(spread) and not np.isnan(bid) and not np.isnan(ask):
        spread = abs(ask - bid)

    return bid, ask, p_mid


def _apply_slippage_static(price: float, side: str, config: BacktestConfig) -> float:
    if np.isnan(price):
        return price
    slippage = config.slippage_bps / 10000.0
    if config.slippage_ticks:
        slippage = max(slippage, config.slippage_ticks)
    if side in {"BUY_YES", "SELL_NO"}:
        return min(1.0, price * (1 + slippage))
    return max(0.0, price * (1 - slippage))


@dataclass
class BacktestConfig:
    threshold: float = 0.05
    slippage_bps: float = 5.0
    slippage_ticks: float = 0.0
    size: float = 1.0
    use_eval_gates: bool = True
    throttle_config: Dict = field(default_factory=dict)
    risk_config: Dict = field(default_factory=dict)
    filter_config: Dict = field(default_factory=dict)
    # Structural arbitrage replay settings
    arb_enabled: bool = False
    arb_edge_min: float = 0.01
    arb_slippage_buffer: float = 0.01
    arb_execution_buffer: float = 0.01
    arb_size: float = 1.0
    cooldown_after_partial_fill_sec: float = 300.0
    arb_max_outcomes: int = 12
    arb_min_depth_per_leg: float = 0.0


@dataclass
class TradeRecord:
    snapshot_ts: pd.Timestamp
    market_id: str
    event_id: str
    side: str
    price: float
    executed_price: float
    size: float
    p_mean: float
    p_lcb: float
    p_ucb: float
    bid: float
    ask: float
    edge: float
    realized_pnl: float


@dataclass
class ArbReplayState:
    attempts: int = 0
    wins: int = 0
    losses: int = 0
    partial_fills: int = 0
    pnl: float = 0.0
    attempt_loss: float = 0.0


def replay_structural_arb(snapshots: pd.DataFrame, config: BacktestConfig) -> ArbReplayState:
    """Simulate structural arbitrage trades from snapshot rows."""

    state = ArbReplayState()
    if not config.arb_enabled:
        return state

    if "snapshot_ts" not in snapshots.columns:
        return state

    df = snapshots.copy()
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], errors="coerce")
    df = df.dropna(subset=["snapshot_ts"])
    if df.empty:
        return state

    df["__event_id"] = df.apply(_extract_event_id, axis=1)
    df["__token_id"] = df.apply(_extract_token_id, axis=1)
    df = df.dropna(subset=["__event_id", "__token_id"])
    if df.empty:
        return state

    cooldowns: Dict[str, float] = {}

    # Ensure chronological processing
    df = df.sort_values("snapshot_ts")

    for (event_id, snapshot_ts), group in df.groupby(["__event_id", "snapshot_ts"], sort=False):
        ts_val = pd.to_datetime(snapshot_ts).timestamp()
        cooldown_until = cooldowns.get(event_id, 0)
        if cooldown_until and cooldown_until > ts_val:
            continue

        legs = []
        for _, row in group.iterrows():
            bid, ask, _ = _price_from_series(row)
            depth = float(row.get("depth", np.nan)) if "depth" in row else np.nan
            if np.isnan(ask):
                continue
            if config.arb_min_depth_per_leg and (np.isnan(depth) or depth < config.arb_min_depth_per_leg):
                legs = []
                break
            legs.append((row["__token_id"], ask, depth))

        if len(legs) < 2 or len(legs) > config.arb_max_outcomes:
            continue

        total_cost = sum(price for _, price, _ in legs)
        edge = 1.0 - total_cost
        effective_edge = edge - (config.arb_slippage_buffer + config.arb_execution_buffer)
        if effective_edge <= config.arb_edge_min:
            continue

        min_depth = min([d for _, _, d in legs if not np.isnan(d)], default=np.nan)
        if np.isnan(min_depth) or min_depth <= 0:
            continue

        desired_size = config.arb_size
        filled_size = min(desired_size, min_depth)
        if filled_size <= 0:
            continue

        exec_cost = sum(_apply_slippage_static(price, "BUY_YES", config) for _, price, _ in legs)
        exec_cost *= filled_size

        pnl = filled_size - exec_cost
        state.attempts += 1
        state.pnl += pnl

        if filled_size < desired_size:
            state.partial_fills += 1
            state.attempt_loss += exec_cost
            cooldowns[event_id] = ts_val + config.cooldown_after_partial_fill_sec
            continue

        if pnl >= 0:
            state.wins += 1
        else:
            state.losses += 1

    return state


class Portfolio:
    def __init__(self, starting_cash: float = 10000.0):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions: Dict[str, Dict[str, float]] = {}
        self.realized_pnl: float = 0.0

    def apply_trade(self, market_id: str, side: str, price: float, size: float) -> float:
        """Update positions and cash, returning realized PnL from this trade."""

        trade_qty = size if side in {"BUY_YES", "SELL_NO"} else -size
        pos = self.positions.get(market_id, {"qty": 0.0, "avg": 0.0})
        qty, avg = pos["qty"], pos["avg"]

        realized = 0.0
        if qty * trade_qty < 0:  # closing existing exposure
            closing_qty = min(abs(qty), abs(trade_qty))
            if qty > 0:
                realized += closing_qty * (price - avg)
            else:
                realized += closing_qty * (avg - price)

        new_qty = qty + trade_qty
        if new_qty != 0:
            if qty * trade_qty >= 0:
                new_avg = (abs(qty) * avg + abs(trade_qty) * price) / abs(new_qty)
            elif abs(trade_qty) > abs(qty):
                new_avg = price
            else:
                new_avg = avg
        else:
            new_avg = 0.0

        self.cash -= trade_qty * price
        self.realized_pnl += realized
        if new_qty == 0:
            self.positions.pop(market_id, None)
        else:
            self.positions[market_id] = {"qty": new_qty, "avg": new_avg}

        return realized

    def mark_to_market(self, marks: Dict[str, float]) -> Tuple[float, float]:
        unrealized = 0.0
        for market_id, pos in self.positions.items():
            mark = marks.get(market_id, 0.5)
            unrealized += pos["qty"] * mark
        equity = self.cash + unrealized
        return equity, unrealized


class BacktestRunner:
    def __init__(
        self,
        snapshots_path: str,
        model_path: str,
        out_dir: str,
        config: Optional[BacktestConfig] = None,
        metrics_logger: Optional[MetricsLogger] = None,
    ):
        self.snapshots_path = Path(snapshots_path)
        self.model_path = Path(model_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if not self.snapshots_path.exists():
            raise FileNotFoundError(f"Snapshots not found at {self.snapshots_path}")

        self.snapshots = pd.read_parquet(self.snapshots_path)
        if "snapshot_ts" not in self.snapshots.columns:
            raise ValueError("snapshots parquet missing 'snapshot_ts'")
        self.snapshots["snapshot_ts"] = pd.to_datetime(
            self.snapshots["snapshot_ts"], errors="coerce"
        )
        self.snapshots = self.snapshots.dropna(subset=["snapshot_ts"])
        self.snapshots = self.snapshots.sort_values("snapshot_ts")

        self.config = config or BacktestConfig()
        self.throttle = TradingBotConfig(**self.config.throttle_config)
        risk_kwargs = {"stale_seconds": float("inf")}
        risk_kwargs.update(self.config.risk_config)
        self.risk = RiskManager(**risk_kwargs)
        self.broker = PaperBroker()
        self.metrics_logger = metrics_logger or MetricsLogger(self.out_dir)
        self.predictor = PolymarketPredictor.load_artifact(str(self.model_path))
        self.predictor.metrics_logger = self.metrics_logger
        self.filter_config = MarketFilterConfig(**self.config.filter_config)

        self.last_eval: Dict[str, Dict[str, float]] = {}
        self.marks: Dict[str, float] = {}
        self.trade_log: List[TradeRecord] = []
        self.equity_curve: List[Dict] = []
        self.arb_metrics: ArbReplayState = ArbReplayState()

    # ------------------------------------------------------------------
    def _price_from_row(self, row: pd.Series) -> Tuple[float, float, float]:
        return _price_from_series(row)

    def _apply_slippage(self, price: float, side: str) -> float:
        if np.isnan(price):
            return price
        slippage = self.config.slippage_bps / 10000.0
        if self.config.slippage_ticks:
            slippage = max(slippage, self.config.slippage_ticks)
        if side in {"BUY_YES", "SELL_NO"}:
            return min(1.0, price * (1 + slippage))
        return max(0.0, price * (1 - slippage))

    def _should_evaluate(self, market_id: str, bid: float, ask: float, ts: float) -> bool:
        if not self.config.use_eval_gates:
            return True
        last = self.last_eval.get(market_id)
        if not last:
            return True
        bid_move = (
            abs(bid - last.get("bid", np.nan))
            if not np.isnan(bid) and not np.isnan(last.get("bid", np.nan))
            else np.inf
        )
        ask_move = (
            abs(ask - last.get("ask", np.nan))
            if not np.isnan(ask) and not np.isnan(last.get("ask", np.nan))
            else np.inf
        )
        tick_move = max(bid_move, ask_move)
        if tick_move >= self.throttle.min_ticks_change_to_recalc:
            return True
        min_interval = (
            1.0 / self.throttle.max_evals_per_market_per_sec
            if self.throttle.max_evals_per_market_per_sec > 0
            else 0
        )
        if ts - last.get("ts", 0) >= min_interval:
            return True
        logger.info("Skip eval %s due to throttle", market_id)
        if self.metrics_logger:
            self.metrics_logger.log_event(
                "health", payload={"reason": "backtest_throttle", "market_id": market_id}
            )
        return False

    def _decision(self, prob: Dict, bid: float, ask: float) -> Optional[Dict]:
        p_lcb, p_ucb = prob.get("p_lcb", 0.5), prob.get("p_ucb", 0.5)
        if not np.isnan(ask) and (p_lcb - ask) > self.config.threshold:
            return {"side": "BUY_YES", "price": ask}
        if not np.isnan(bid) and (bid - p_ucb) > self.config.threshold:
            return {"side": "SELL_YES", "price": bid}
        return None

    def _feature_frame(self, row: pd.Series, price_hint: float) -> pd.DataFrame:
        feature_dict = row.to_dict()
        feature_dict.setdefault("p_mid", price_hint)
        df = pd.DataFrame([feature_dict])
        return df

    def _build_market_state(self, row: pd.Series) -> Dict:
        return {
            "bid": float(row.get("bid", np.nan)),
            "ask": float(row.get("ask", np.nan)),
            "mid": float(row.get("p_mid", np.nan)),
            "spread": float(row.get("spread", np.nan)),
            "depth": float(row.get("depth", np.nan)) if "depth" in row else np.nan,
            "volume24h": float(row.get("volume24h", np.nan)) if "volume24h" in row else np.nan,
            "trades_last_24h": row.get("trades_last_24h"),
            "time_to_resolve": float(row.get("time_to_resolve", np.nan)) if "time_to_resolve" in row else None,
        }

    def _log_equity(self, snapshot_ts: pd.Timestamp, portfolio: Portfolio):
        equity, unrealized = portfolio.mark_to_market(self.marks)
        record = {
            "snapshot_ts": snapshot_ts,
            "equity": equity,
            "cash": portfolio.cash,
            "realized_pnl": portfolio.realized_pnl,
            "unrealized_pnl": unrealized,
        }
        self.equity_curve.append(record)
        if self.metrics_logger:
            self.metrics_logger.log_event(
                "health",
                payload={
                    "timestamp": snapshot_ts.timestamp(),
                    "equity": equity,
                    "cash": portfolio.cash,
                    "unrealized": unrealized,
                },
            )
        return equity

    def run(self) -> Dict:
        if self.config.arb_enabled:
            self.arb_metrics = replay_structural_arb(self.snapshots, self.config)

        portfolio = Portfolio()
        last_equity = portfolio.cash

        for _, row in self.snapshots.iterrows():
            market_id = row.get("market_id") or row.get("id")
            if not market_id:
                continue
            event_id = row.get("event_id") or market_id
            snapshot_ts = pd.to_datetime(row["snapshot_ts"])
            ts = snapshot_ts.timestamp()

            bid, ask, p_mid = self._price_from_row(row)
            self.marks[market_id] = p_mid if not np.isnan(p_mid) else 0.5

            market_state = self._build_market_state(row)
            ok, reasons = should_trade_market(
                market_state,
                row.to_dict(),
                now_ts=ts,
                config=self.filter_config,
            )
            if not ok:
                if self.metrics_logger:
                    self.metrics_logger.log_event(
                        "filter_skip", payload={"market_id": market_id, "reasons": reasons}
                    )
                logger.info("Skip %s due to filters: %s", market_id, ",".join(reasons))
                equity = self._log_equity(snapshot_ts, portfolio)
                self.risk.mark_pnl(equity - last_equity)
                last_equity = equity
                continue

            if not self._should_evaluate(market_id, bid, ask, ts):
                equity = self._log_equity(snapshot_ts, portfolio)
                self.risk.mark_pnl(equity - last_equity)
                last_equity = equity
                continue

            feature_df = self._feature_frame(row, p_mid)
            prob = self.predictor.predict_probability_from_features(
                feature_df, price_hint=p_mid, market_id=market_id
            )
            decision = self._decision(prob, bid, ask)
            if decision:
                notional = decision["price"] * self.config.size
                if self.metrics_logger:
                    edge_val = prob.get("p_lcb") - decision["price"] if decision["side"] == "BUY_YES" else decision["price"] - prob.get("p_ucb")
                    self.metrics_logger.log_event(
                        "decision",
                        payload={
                            "market_id": market_id,
                            "action": decision["side"],
                            "size": self.config.size,
                            "threshold": self.config.threshold,
                            "edge": edge_val,
                        },
                    )
                if self.risk.can_trade(market_id, event_id, notional, time.time()):
                    exec_price = self._apply_slippage(decision["price"], decision["side"])
                    realized = portfolio.apply_trade(
                        market_id, decision["side"], exec_price, self.config.size
                    )
                    self.risk.record_trade(market_id, event_id, decision["side"], notional)
                    edge = (
                        prob["p_lcb"] - exec_price
                        if decision["side"] == "BUY_YES"
                        else exec_price - prob["p_ucb"]
                    )
                    trade_record = TradeRecord(
                        snapshot_ts=snapshot_ts,
                        market_id=market_id,
                        event_id=event_id,
                        side=decision["side"],
                        price=decision["price"],
                        executed_price=exec_price,
                        size=self.config.size,
                        p_mean=prob.get("p_mean", np.nan),
                        p_lcb=prob.get("p_lcb", np.nan),
                        p_ucb=prob.get("p_ucb", np.nan),
                        bid=bid,
                        ask=ask,
                        edge=edge,
                        realized_pnl=realized,
                    )
                    self.trade_log.append(trade_record)
                    if self.metrics_logger:
                        self.metrics_logger.log_event(
                            "trade",
                            payload={
                                "market_id": market_id,
                                "action": decision["side"],
                                "price": exec_price,
                                "size": self.config.size,
                                "pnl": realized,
                            },
                        )
                elif self.metrics_logger:
                    self.metrics_logger.log_event(
                        "decision",
                        payload={
                            "market_id": market_id,
                            "action": "blocked_risk",
                            "threshold": self.config.threshold,
                        },
                    )
            equity = self._log_equity(snapshot_ts, portfolio)
            self.risk.mark_pnl(equity - last_equity)
            last_equity = equity
            self.last_eval[market_id] = {"bid": bid, "ask": ask, "ts": ts}

        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame([t.__dict__ for t in self.trade_log])

        summary = self._summarize(equity_df, trades_df, portfolio)
        self._write_outputs(equity_df, trades_df, summary)
        return summary

    def _summarize(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame, portfolio: Portfolio) -> Dict:
        starting_equity = portfolio.starting_cash
        final_equity = equity_df["equity"].iloc[-1] if not equity_df.empty else starting_equity
        total_pnl = final_equity - starting_equity
        max_equity = equity_df["equity"].cummax() if not equity_df.empty else pd.Series([starting_equity])
        drawdown = (equity_df["equity"] - max_equity) if not equity_df.empty else pd.Series([0])
        max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

        per_market = {}
        for market_id, pos in portfolio.positions.items():
            per_market[market_id] = float(pos["qty"] * self.marks.get(market_id, 0.5))

        hit_rate = 0.0
        avg_edge = 0.0
        turnover = 0.0
        if not trades_df.empty:
            hit_rate = float((trades_df["realized_pnl"] > 0).mean())
            avg_edge = float(trades_df["edge"].mean())
            turnover = float((trades_df["executed_price"] * trades_df["size"]).abs().sum())

        return {
            "total_pnl": float(total_pnl),
            "max_drawdown": float(max_dd),
            "final_equity": float(final_equity),
            "turnover": float(turnover),
            "hit_rate": float(hit_rate),
            "avg_edge": float(avg_edge),
            "per_market_unrealized": per_market,
            "arb_metrics": self.arb_metrics.__dict__ if self.arb_metrics else {},
        }

    def _write_outputs(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame, summary: Dict):
        equity_path = self.out_dir / "equity_curve.csv"
        trades_path = self.out_dir / "trades.csv"
        summary_path = self.out_dir / "summary.json"

        equity_df.to_csv(equity_path, index=False)
        trades_df.to_csv(trades_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2))

        try:  # optional parquet outputs
            equity_df.to_parquet(self.out_dir / "equity_curve.parquet", index=False)
            trades_df.to_parquet(self.out_dir / "trades.parquet", index=False)
        except Exception:
            logger.info("Parquet outputs skipped (engine missing)")


def main():  # pragma: no cover - CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Replay backtester over snapshots")
    parser.add_argument("--snapshots", required=True, help="Path to snapshots parquet")
    parser.add_argument("--model", required=True, help="Path to trained model artifact")
    parser.add_argument("--out", required=True, help="Directory for results")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--slippage_bps", type=float, default=5.0)
    parser.add_argument("--slippage_ticks", type=float, default=0.0)
    parser.add_argument("--size", type=float, default=1.0)
    parser.add_argument("--disable_gates", action="store_true", help="Disable eval throttles")

    args = parser.parse_args()
    config = BacktestConfig(
        threshold=args.threshold,
        slippage_bps=args.slippage_bps,
        slippage_ticks=args.slippage_ticks,
        size=args.size,
        use_eval_gates=not args.disable_gates,
    )

    runner = BacktestRunner(
        snapshots_path=args.snapshots,
        model_path=args.model,
        out_dir=args.out,
        config=config,
    )
    summary = runner.run()
    logger.info("Backtest complete: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
