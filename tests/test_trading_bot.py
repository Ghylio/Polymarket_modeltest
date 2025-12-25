import time
import types

import pandas as pd

from trading_bot import LiveMarketState, ProbabilityTradingBot, RiskManager


class FakeFetcher:
    def __init__(self, bid=0.45, ask=0.55):
        self.bid = bid
        self.ask = ask

    def get_token_ids_for_market(self, market):
        return "YES", "NO"

    def get_orderbook(self, token_id):  # noqa: ARG002
        return {"bids": [[self.bid, 100]], "asks": [[self.ask, 100]]}

    def get_trades(self, token_id, limit=200):  # noqa: ARG002
        ts = pd.Timestamp.utcnow()
        return [
            {"price": self.bid, "size": 10, "timestamp": ts.isoformat()},
            {"price": self.ask, "size": 5, "timestamp": ts.isoformat()},
        ]

    def trades_to_dataframe(self, trades):
        return pd.DataFrame(trades)

    def get_markets(self, limit=10, order="volume24hr"):  # noqa: ARG002
        return [
            {"id": "m1", "eventId": "e1", "outcomePrices": [0.5, 0.5]},
            {"id": "m2", "eventId": "e1", "outcomePrices": [0.5, 0.5]},
        ][:limit]


class FixedProbabilityPredictor:
    def predict_probability(self, market, trades_df):  # noqa: ARG002
        return {
            "p_mean": 0.65,
            "p_lcb": 0.60,
            "p_ucb": 0.70,
            "trade_features": {},
            "market_features": {},
        }


def test_trade_triggers_on_edge():
    fetcher = FakeFetcher(bid=0.48, ask=0.50)
    bot = ProbabilityTradingBot(fetcher=fetcher, threshold=0.05)
    bot.predictor = FixedProbabilityPredictor()

    trades = bot.run_once(limit=1)
    assert trades, "Expected a trade when p_lcb exceeds ask by threshold"
    assert trades[0]["side"] == "BUY_YES"


def test_risk_manager_blocks_over_cap():
    rm = RiskManager(per_market_cap=1.0, per_event_cap=1.0, daily_loss_limit=10.0, stale_seconds=60)
    now = time.time()
    assert rm.can_trade("m", "e", 0.5, now)
    rm.record_trade("m", "e", "BUY_YES", 0.5)
    assert not rm.can_trade("m", "e", 0.6, now)


def test_stale_data_blocks_trade():
    state = LiveMarketState(bid=0.4, ask=0.6, mid=0.5, spread=0.2, depth=10, last_update=time.time() - 400)
    bot = ProbabilityTradingBot(fetcher=FakeFetcher(), threshold=0.01)
    bot.live_state["m1"] = state
    bot.predictor = FixedProbabilityPredictor()

    result = bot.process_market({"id": "m1", "eventId": "e1", "outcomePrices": [0.5, 0.5]})
    assert result is None
