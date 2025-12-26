import time
import unittest

from bot.market_filters import MarketFilterConfig
from bot.strategy_structural_arb import StructuralArbConfig, StructuralArbStrategy
from negrisk_arb import OrderExecutor


class DummyExecutor(OrderExecutor):
    def __init__(self, fills):
        self.fills = fills
        self.unwound = False
        self.orders = None
        self.last_tif = None

    def place_batch_orders(self, orders, time_in_force: str = "FOK"):
        self.orders = orders
        self.last_tif = time_in_force
        return self.fills

    def unwind(self, fills):
        self.unwound = True
        return fills


class FakeFetcher:
    def __init__(self, orderbooks, mappings):
        self.orderbooks = orderbooks
        self.mappings = mappings

    def get_outcome_token_map(self, market):
        return self.mappings.get(market["id"])

    def get_orderbook(self, token_id, depth=None):  # noqa: ARG002
        return self.orderbooks.get(token_id, {"asks": []})


class StructuralArbStrategyTest(unittest.TestCase):
    def test_edge_and_size_from_depth(self):
        markets = [{"id": "m1", "eventId": "e1", "rules": "valid rules" * 10, "category": "negRisk"}]
        mapping = {
            "m1": {
                "outcome_token_map": {"A": "t1", "B": "t2", "C": "t3"},
            }
        }
        orderbooks = {
            "t1": {"asks": [[0.2, 5.0]]},
            "t2": {"asks": [[0.3, 2.0]]},
            "t3": {"asks": [[0.4, 3.0]]},
        }
        fetcher = FakeFetcher(orderbooks, mapping)
        config = StructuralArbConfig(
            enabled=True,
            allow_non_negrisk=True,
            min_edge_abs=0.01,
            slippage_bps=0.0,
            fee_bps=0.0,
            min_depth_per_leg_usdc=0.0,
            min_depth_within_band_usdc=0.0,
            depth_robustness_enabled=False,
        )
        fills = [
            {"token_id": "t1", "price": 0.2, "size": 2.0, "filled_size": 2.0},
            {"token_id": "t2", "price": 0.3, "size": 2.0, "filled_size": 2.0},
            {"token_id": "t3", "price": 0.4, "size": 2.0, "filled_size": 2.0},
        ]
        executor = DummyExecutor(fills)
        strategy = StructuralArbStrategy(fetcher=fetcher, executor=executor, config=config, filter_config=MarketFilterConfig())

        result = strategy.evaluate_event_group(markets)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["edge"], 0.1)
        self.assertAlmostEqual(result["size"], 2.0)
        self.assertEqual(executor.last_tif, "FOK")

    def test_size_limited_by_caps(self):
        markets = [{"id": "m1", "eventId": "e1", "rules": "valid rules" * 10, "category": "negRisk"}]
        mapping = {"m1": {"outcome_token_map": {"A": "t1", "B": "t2", "C": "t3"}}}
        orderbooks = {
            "t1": {"asks": [[0.2, 10.0]]},
            "t2": {"asks": [[0.3, 10.0]]},
            "t3": {"asks": [[0.4, 10.0]]},
        }
        fetcher = FakeFetcher(orderbooks, mapping)
        config = StructuralArbConfig(
            enabled=True,
            allow_non_negrisk=True,
            max_usdc_per_event=1.0,
            max_usdc_per_cycle=0.5,
            min_edge_abs=0.01,
            slippage_bps=0.0,
            fee_bps=0.0,
            min_depth_per_leg_usdc=0.0,
            min_depth_within_band_usdc=0.0,
            depth_robustness_enabled=False,
        )
        fills = [
            {"token_id": "t1", "price": 0.2, "size": 1.0, "filled_size": 1.0},
            {"token_id": "t2", "price": 0.3, "size": 1.0, "filled_size": 1.0},
            {"token_id": "t3", "price": 0.4, "size": 1.0, "filled_size": 1.0},
        ]
        executor = DummyExecutor(fills)
        strategy = StructuralArbStrategy(fetcher=fetcher, executor=executor, config=config, filter_config=MarketFilterConfig())

        result = strategy.evaluate_event_group(markets)
        self.assertIsNotNone(result)
        # max_usdc_per_cycle binds: 0.5 / 0.9 ≈ 0.5556
        self.assertAlmostEqual(result["size"], 0.5 / 0.9, places=4)

    def test_partial_fill_triggers_cooldown_and_attempt_loss(self):
        markets = [{"id": "m1", "eventId": "e1", "category": "negRisk", "rules": "clear rules" * 10}]
        mapping = {"m1": {"outcome_token_map": {"A": "t1", "B": "t2", "C": "t3"}}}
        orderbooks = {
            "t1": {"asks": [[0.2, 1.0]]},
            "t2": {"asks": [[0.2, 1.0]]},
            "t3": {"asks": [[0.2, 1.0]]},
        }
        fetcher = FakeFetcher(orderbooks, mapping)
        config = StructuralArbConfig(
            enabled=True,
            allow_non_negrisk=True,
            slippage_bps=0.0,
            fee_bps=0.0,
            cooldown_after_partial_fill_sec=10,
            min_depth_per_leg_usdc=0.0,
            min_depth_within_band_usdc=0.0,
            depth_robustness_enabled=False,
        )
        partial_fills = [
            {"token_id": "t1", "price": 0.2, "size": 1.0, "filled_size": 0.5},
            {"token_id": "t2", "price": 0.2, "size": 1.0, "filled_size": 1.0},
            {"token_id": "t3", "price": 0.2, "size": 1.0, "filled_size": 1.0},
        ]
        executor = DummyExecutor(partial_fills)
        strategy = StructuralArbStrategy(fetcher=fetcher, executor=executor, config=config, filter_config=MarketFilterConfig())

        first = strategy.evaluate_event_group(markets)
        self.assertIsNone(first)
        self.assertGreater(strategy.cooldowns.get("e1", 0), time.time())
        self.assertTrue(executor.unwound)
        self.assertGreater(strategy.risk.arb_tracker.attempt_loss, 0)

        second = strategy.evaluate_event_group(markets)
        self.assertIsNone(second)

    def test_filters_and_batch_limit(self):
        markets = [{"id": "m1", "eventId": "e1", "category": "negRisk", "volume24hr": 0.0, "rules": "clear rules" * 10}]
        mapping = {"m1": {"outcome_token_map": {"A": "t1", "B": "t2", "C": "t3"}}}
        orderbooks = {
            "t1": {"asks": [[0.6, 1.0]]},
            "t2": {"asks": [[0.2, 1.0]]},
            "t3": {"asks": [[0.2, 1.0]]},
        }
        fetcher = FakeFetcher(orderbooks, mapping)
        config = StructuralArbConfig(
            enabled=True,
            allow_non_negrisk=True,
            slippage_bps=0.0,
            fee_bps=0.0,
            min_edge_abs=0.0,
            min_depth_per_leg_usdc=0.0,
            max_batches_per_minute=1,
        )
        filter_config = MarketFilterConfig(min_top_of_book_depth=2.0)
        fills = [
            {"token_id": "t1", "price": 0.6, "size": 1.0, "filled_size": 1.0},
            {"token_id": "t2", "price": 0.2, "size": 1.0, "filled_size": 1.0},
            {"token_id": "t3", "price": 0.2, "size": 1.0, "filled_size": 1.0},
        ]
        executor = DummyExecutor(fills)
        strategy = StructuralArbStrategy(fetcher=fetcher, executor=executor, config=config, filter_config=filter_config)

        # Filter blocks first attempt due to depth
        result = strategy.evaluate_event_group(markets)
        self.assertIsNone(result)

        # Second attempt should be skipped due to batch limiter even if filters would pass
        result = strategy.evaluate_event_group(markets)
        self.assertIsNone(result)

    def test_depth_within_band_drives_sizing(self):
        markets = [{"id": "m1", "eventId": "e1", "category": "negRisk", "rules": "clear rules" * 10}]
        mapping = {"m1": {"outcome_token_map": {"A": "t1", "B": "t2", "C": "t3"}}}
        orderbooks = {
            "t1": {"asks": [[0.2, 0.5], [0.21, 10.0]]},
            "t2": {"asks": [[0.3, 0.5], [0.31, 10.0]]},
            "t3": {"asks": [[0.4, 0.5], [0.41, 10.0]]},
        }
        fetcher = FakeFetcher(orderbooks, mapping)
        config = StructuralArbConfig(
            enabled=True,
            allow_non_negrisk=True,
            slippage_bps=0.0,
            fee_bps=0.0,
            min_edge_abs=0.0,
            min_depth_per_leg_usdc=0.0,
            min_depth_within_band_usdc=2.0,
            depth_robustness_ticks=1,
        )
        fills = [
            {"token_id": "t1", "price": 0.21, "size": 5.0, "filled_size": 5.0},
            {"token_id": "t2", "price": 0.31, "size": 5.0, "filled_size": 5.0},
            {"token_id": "t3", "price": 0.41, "size": 5.0, "filled_size": 5.0},
        ]
        executor = DummyExecutor(fills)
        strategy = StructuralArbStrategy(fetcher=fetcher, executor=executor, config=config, filter_config=MarketFilterConfig())

        result = strategy.evaluate_event_group(markets)
        self.assertIsNotNone(result)
        # Robust depth (0.5+10) on each leg supports larger size even though best ask size is 0.5
        self.assertGreater(result["size"], 0.5)

    def test_insufficient_depth_within_band_blocks(self):
        markets = [{"id": "m1", "eventId": "e1", "category": "negRisk", "rules": "clear rules" * 10}]
        mapping = {"m1": {"outcome_token_map": {"A": "t1", "B": "t2", "C": "t3"}}}
        orderbooks = {
            "t1": {"asks": [[0.2, 0.5]]},
            "t2": {"asks": [[0.3, 0.5]]},
            "t3": {"asks": [[0.4, 0.5], [0.41, 0.1]]},
        }
        fetcher = FakeFetcher(orderbooks, mapping)
        config = StructuralArbConfig(
            enabled=True,
            allow_non_negrisk=True,
            slippage_bps=0.0,
            fee_bps=0.0,
            min_edge_abs=0.0,
            min_depth_per_leg_usdc=0.0,
            min_depth_within_band_usdc=1.0,
            depth_robustness_ticks=1,
        )
        fills = [
            {"token_id": "t1", "price": 0.2, "size": 0.5, "filled_size": 0.5},
            {"token_id": "t2", "price": 0.3, "size": 0.5, "filled_size": 0.5},
            {"token_id": "t3", "price": 0.4, "size": 0.5, "filled_size": 0.5},
        ]
        executor = DummyExecutor(fills)
        strategy = StructuralArbStrategy(fetcher=fetcher, executor=executor, config=config, filter_config=MarketFilterConfig())

        result = strategy.evaluate_event_group(markets)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
