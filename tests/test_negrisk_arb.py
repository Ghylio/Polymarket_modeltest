import unittest

from negrisk_arb import NegRiskArbitrageur, NegRiskConfig, PaperBatchExecutor
from trading_bot import RiskManager


class FakeFetcher:
    def __init__(self, asks):
        # asks: dict token -> best ask
        self.asks = asks

    def get_orderbook(self, token_id):
        price = self.asks[token_id]
        return {"asks": [[price, 100]], "bids": [[price - 0.1, 50]]}


class PartialExecutor(PaperBatchExecutor):
    def __init__(self):
        super().__init__()
        self.unwind_called = False

    def place_batch_orders(self, orders, time_in_force="FOK"):
        fills = super().place_batch_orders(orders, time_in_force=time_in_force)
        if fills:
            # mark the last leg as partial
            fills[-1]["filled_size"] = fills[-1]["size"] / 2
            fills[-1]["status"] = "partial"
        return fills

    def unwind(self, fills):  # noqa: ARG002
        self.unwind_called = True
        return []


class NegRiskArbTests(unittest.TestCase):
    def test_full_set_cost_and_trigger(self):
        market = {
            "id": "m1",
            "eventId": "e1",
            "clobTokenIds": ["t1", "t2", "t3"],
            "type": "negRisk",
            "rules": "clear resolution criteria" * 3,
        }
        fetcher = FakeFetcher({"t1": 0.3, "t2": 0.25, "t3": 0.35})
        arb = NegRiskArbitrageur(
            fetcher=fetcher,
            risk_manager=RiskManager(per_market_cap=10, per_event_cap=10, daily_loss_limit=10),
            config=NegRiskConfig(buffer=0.05, size_per_outcome=1.0),
            apply_filters=False,
        )

        cost, prices = arb.compute_full_set_cost(market)
        self.assertAlmostEqual(cost, 0.9)
        self.assertEqual(len(prices), 3)

        result = arb.evaluate_and_trade(market)
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "filled")

    def test_partial_fill_triggers_unwind(self):
        market = {
            "id": "m2",
            "eventId": "e2",
            "clobTokenIds": ["t1", "t2"],
            "type": "NEG_RISK",
            "rules": "clear resolution criteria" * 3,
        }
        fetcher = FakeFetcher({"t1": 0.4, "t2": 0.35})
        executor = PartialExecutor()
        arb = NegRiskArbitrageur(
            fetcher=fetcher,
            executor=executor,
            risk_manager=RiskManager(per_market_cap=10, per_event_cap=10, daily_loss_limit=10),
            config=NegRiskConfig(buffer=0.0, size_per_outcome=1.0),
            apply_filters=False,
        )

        result = arb.evaluate_and_trade(market)
        self.assertEqual(result["status"], "unwound")
        self.assertTrue(executor.unwind_called)


if __name__ == "__main__":
    unittest.main()
