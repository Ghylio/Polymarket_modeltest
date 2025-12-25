import time
import unittest

from bot.market_filters import MarketFilterConfig, should_trade_market


class MarketFilterTests(unittest.TestCase):
    def test_volume_and_spread_filters(self):
        cfg = MarketFilterConfig(min_24h_volume=100, max_spread_abs=0.05, max_spread_pct=0.2)
        state = {"bid": 0.45, "ask": 0.55, "spread": 0.1, "volume24h": 50}
        ok, reasons = should_trade_market(state, {}, config=cfg)
        self.assertFalse(ok)
        self.assertIn("low_volume", reasons)
        self.assertIn("wide_spread_abs", reasons)

    def test_depth_and_trades_filters(self):
        cfg = MarketFilterConfig(min_top_of_book_depth=10, min_trades_last_24h=2)
        state = {"depth": 5, "trades_last_24h": 1}
        ok, reasons = should_trade_market(state, {}, config=cfg)
        self.assertFalse(ok)
        self.assertIn("shallow_book", reasons)
        self.assertIn("inactive_trades", reasons)

    def test_rules_and_time_filters(self):
        now = time.time()
        cfg = MarketFilterConfig(skip_if_rules_ambiguous=True, rules_min_length=10, skip_if_time_to_resolve_lt=60)
        market = {"rules": "subjective", "endTime": now + 30}
        ok, reasons = should_trade_market({}, market, now_ts=now, config=cfg)
        self.assertFalse(ok)
        self.assertIn("ambiguous_rules", reasons)
        self.assertIn("too_close_to_resolution", reasons)

    def test_passes_filters(self):
        now = time.time()
        cfg = MarketFilterConfig(
            min_24h_volume=10,
            max_spread_abs=0.1,
            min_top_of_book_depth=1,
            min_trades_last_24h=0,
            skip_if_time_to_resolve_lt=0,
        )
        state = {
            "bid": 0.48,
            "ask": 0.5,
            "spread": 0.02,
            "volume24h": 20,
            "depth": 5,
            "trades_last_24h": 5,
            "time_to_resolve": 120,
        }
        ok, reasons = should_trade_market(state, {"rules": "clear market rules" * 3}, now_ts=now, config=cfg)
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_missing_metrics_reported(self):
        cfg = MarketFilterConfig(min_24h_volume=1, min_top_of_book_depth=1, min_trades_last_24h=1)
        ok, reasons = should_trade_market({}, {}, config=cfg)
        self.assertFalse(ok)
        self.assertIn("missing_volume", reasons)
        self.assertIn("missing_depth", reasons)
        self.assertIn("missing_trades", reasons)


if __name__ == "__main__":
    unittest.main()
