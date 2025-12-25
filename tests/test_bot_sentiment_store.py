import time
from datetime import datetime

import numpy as np
import pandas as pd

from sentiment.store import DocumentStore
from trading_bot import LiveMarketState, ProbabilityTradingBot


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
        ][:limit]


def _basic_state(bid=0.45, ask=0.55):
    return LiveMarketState(bid=bid, ask=ask, mid=(bid + ask) / 2, spread=abs(ask - bid), depth=10, last_update=time.time())


def _dummy_trades():
    ts = pd.Timestamp.utcnow()
    return pd.DataFrame({"price": [0.5, 0.5], "size": [1.0, 1.0], "timestamp": [ts, ts]})


def test_bot_uses_sentiment_store(tmp_path):
    db_path = tmp_path / "sent.db"
    store = DocumentStore(db_path)
    as_of = datetime.utcnow()
    bucket_ts = int(as_of.replace(minute=0, second=0, microsecond=0).timestamp())
    store.upsert_aggregate(
        market_id="m1",
        bucket_ts=bucket_ts,
        agg={
            "sent_mean_1h": 0.1,
            "sent_std_1h": 0.0,
            "doc_count_1h": 2,
            "sent_mean_6h": 0.2,
            "sent_std_6h": 0.0,
            "doc_count_6h": 2,
            "sent_mean_24h": 0.3,
            "sent_std_24h": 0.0,
            "doc_count_24h": 2,
            "sent_mean_7d": 0.4,
            "sent_std_7d": 0.0,
            "doc_count_7d": 2,
            "sent_trend": -0.2,
        },
    )

    bot = ProbabilityTradingBot(fetcher=FakeFetcher(), threshold=1.0, sentiment_store=store)
    state = _basic_state()
    trades_df = _dummy_trades()
    prob = bot.predict_market_prob(
        {"id": "m1", "eventId": "e1", "outcomePrices": [0.5, 0.5]},
        token_id="YES",
        state=state,
        trades_df=trades_df,
    )

    feature_df = prob.get("features")
    assert not feature_df.empty
    assert np.isclose(feature_df["sent_mean_1h"].iloc[0], 0.1)
    assert np.isclose(feature_df["sent_trend"].iloc[0], -0.2)


def test_bot_handles_missing_sentiment_store(tmp_path):
    db_path = tmp_path / "missing.db"
    # Do not insert aggregates
    store = DocumentStore(db_path)

    bot = ProbabilityTradingBot(fetcher=FakeFetcher(), threshold=1.0, sentiment_store=store)
    state = _basic_state()
    prob = bot.predict_market_prob(
        {"id": "m2", "eventId": "e2", "outcomePrices": [0.5, 0.5]},
        token_id="YES",
        state=state,
        trades_df=_dummy_trades(),
    )

    feature_df = prob.get("features")
    assert feature_df is not None
    assert np.isnan(feature_df["sent_mean_1h"].iloc[0])
    assert np.isnan(feature_df["sent_trend"].iloc[0])
