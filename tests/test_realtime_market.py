import time
from pathlib import Path
from unittest import mock

import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realtime_market import RealtimeMarketStream
from trading_bot import LiveMarketState, ProbabilityTradingBot, TradingBotConfig


def test_realtime_reconnects_after_close():
    attempts = []

    class DummyWS:
        def __init__(self, url, on_message=None, on_error=None, on_close=None, on_open=None):
            attempts.append("connect")
            self.on_close = on_close
            self.on_open = on_open

        def run_forever(self, **kwargs):
            # Simulate immediate open and close
            if self.on_open:
                self.on_open(self)
            if self.on_close:
                self.on_close(self)

        def close(self):
            pass

        def send(self, msg):
            self.last_msg = msg

    stream = RealtimeMarketStream(
        ["token-1"],
        on_book=lambda update: None,
        ws_factory=lambda *args, **kwargs: DummyWS(*args, **kwargs),
        reconnect_delay=0.01,
    )
    stream.start()
    time.sleep(0.05)
    stream.stop()
    assert len(attempts) >= 2  # initial + reconnect attempt


def test_stale_data_pause_triggers_without_realtime():
    class StubFetcher:
        def get_token_ids_for_market(self, market):
            return market["clobTokenIds"][0], None

        def get_orderbook(self, token_id, depth=None):
            return {"bids": [[0.4, 5]], "asks": [[0.6, 5]]}

        def get_trades(self, token_id, limit=200):
            return [
                {
                    "price": 0.5,
                    "size": 1.0,
                    "timestamp": pd.Timestamp.utcnow() - pd.Timedelta(seconds=10),
                }
            ]

        def trades_to_dataframe(self, trades):
            return pd.DataFrame(trades)

        def get_markets(self, limit=10, order="volume24hr"):
            return []

    bot = ProbabilityTradingBot(
        fetcher=StubFetcher(),
        bot_config=TradingBotConfig(stale_book_timeout_sec=1, stale_trade_timeout_sec=1),
        paper_trading=True,
        enable_realtime=False,
        metrics_logger=mock.Mock(),
    )

    market = {"id": "m1", "clobTokenIds": ["token-1"], "volume24hr": 100}
    bot.live_state["m1"] = LiveMarketState(bid=0.4, ask=0.6, mid=0.5, spread=0.2, depth=10, last_update=time.time() - 5)
    result = bot.process_market(market)
    assert result is None
