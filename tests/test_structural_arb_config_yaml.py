import tempfile
import unittest
from pathlib import Path

import yaml

from trading_bot import StructuralArbConfig, TradingBotConfig, load_trading_bot_config


class StructuralArbConfigLoadTest(unittest.TestCase):
    def test_loads_structural_arb_defaults(self):
        cfg = load_trading_bot_config(Path("config/trading_bot_config.yaml"))
        self.assertIsInstance(cfg, TradingBotConfig)
        arb = cfg.structural_arb
        self.assertFalse(arb.enabled)
        self.assertEqual(arb.max_legs_per_event, 15)
        self.assertAlmostEqual(arb.min_edge_abs, 0.02)
        self.assertAlmostEqual(arb.edge_buffer(), 0.001)
        self.assertEqual(arb.max_attempt_loss_per_day, 50)
        self.assertTrue(arb.depth_robustness_enabled)
        self.assertEqual(arb.depth_robustness_ticks, 1)
        self.assertAlmostEqual(arb.depth_robustness_max_extra_cost_abs, 0.01)

    def test_validation_rejects_too_many_legs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "structural_arb": {
                            "enabled": True,
                            "max_legs_per_event": 20,
                        }
                    }
                )
            )
            with self.assertRaises(ValueError):
                load_trading_bot_config(path)


if __name__ == "__main__":
    unittest.main()
