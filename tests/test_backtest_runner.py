import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import pandas as pd

from backtest.run import BacktestConfig, BacktestRunner

try:
    import pyarrow  # noqa: F401

    HAS_PARQUET = True
except ImportError:  # pragma: no cover - environment without parquet engine
    HAS_PARQUET = False


class _FakePredictor:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0
        self.feature_columns = []
        self.is_trained = True

    @classmethod
    def load_artifact(cls, path):  # pragma: no cover - patched in tests
        raise RuntimeError("should be patched")

    def predict_probability_from_features(self, feature_df, price_hint=None, n_obs=50, market_id=None):
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[idx]


@unittest.skipUnless(HAS_PARQUET, "pyarrow required for backtest tests")
class BacktestRunnerTests(unittest.TestCase):
    def _write_snapshots(self, rows, tmpdir: Path) -> Path:
        df = pd.DataFrame(rows)
        path = tmpdir / "snap.parquet"
        df.to_parquet(path)
        return path

    def test_backtest_executes_trades_with_slippage_and_logs(self):
        base = datetime(2024, 1, 1)
        rows = [
            {
                "market_id": "m1",
                "event_id": "e1",
                "snapshot_ts": base,
                "p_mid": 0.5,
                "bid": 0.48,
                "ask": 0.5,
                "spread": 0.02,
                "y": 1,
            },
            {
                "market_id": "m1",
                "event_id": "e1",
                "snapshot_ts": base + timedelta(hours=1),
                "p_mid": 0.55,
                "bid": 0.55,
                "ask": 0.56,
                "spread": 0.01,
                "y": 1,
            },
        ]

        responses = [
            {"p_mean": 0.7, "p_lcb": 0.65, "p_ucb": 0.75},
            {"p_mean": 0.3, "p_lcb": 0.25, "p_ucb": 0.35},
        ]
        fake_predictor = _FakePredictor(responses)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            snap_path = self._write_snapshots(rows, tmp_path)
            out_dir = tmp_path / "out"

            with mock.patch("backtest.run.PolymarketPredictor.load_artifact", return_value=fake_predictor):
                runner = BacktestRunner(
                    snapshots_path=snap_path,
                    model_path=tmp_path / "model.joblib",
                    out_dir=out_dir,
                    config=BacktestConfig(threshold=0.05, slippage_bps=100.0, size=1.0),
                )
                summary = runner.run()

            trades = pd.read_csv(out_dir / "trades.csv")
            self.assertEqual(len(trades), 2)
            self.assertAlmostEqual(trades.iloc[0]["executed_price"], rows[0]["ask"] * 1.01, places=4)
            self.assertAlmostEqual(trades.iloc[1]["executed_price"], rows[1]["bid"] * 0.99, places=4)
            self.assertGreater(summary["total_pnl"], 0)

    def test_risk_limits_block_after_daily_loss(self):
        base = datetime(2024, 1, 1)
        rows = [
            {"market_id": "m2", "snapshot_ts": base, "p_mid": 0.6, "bid": 0.6, "ask": 0.6, "spread": 0.0, "y": 1},
            {"market_id": "m2", "snapshot_ts": base + timedelta(hours=1), "p_mid": 0.1, "bid": 0.1, "ask": 0.1, "spread": 0.0, "y": 1},
            {"market_id": "m2", "snapshot_ts": base + timedelta(hours=2), "p_mid": 0.2, "bid": 0.2, "ask": 0.2, "spread": 0.0, "y": 1},
        ]
        responses = [
            {"p_mean": 0.8, "p_lcb": 0.75, "p_ucb": 0.85},
            {"p_mean": 0.1, "p_lcb": 0.05, "p_ucb": 0.15},
            {"p_mean": 0.8, "p_lcb": 0.75, "p_ucb": 0.85},
        ]
        fake_predictor = _FakePredictor(responses)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            snap_path = self._write_snapshots(rows, tmp_path)
            out_dir = tmp_path / "out"

            with mock.patch("backtest.run.PolymarketPredictor.load_artifact", return_value=fake_predictor):
                runner = BacktestRunner(
                    snapshots_path=snap_path,
                    model_path=tmp_path / "model.joblib",
                    out_dir=out_dir,
                    config=BacktestConfig(
                        threshold=0.0,
                        slippage_bps=0.0,
                        size=1.0,
                        risk_config={"daily_loss_limit": 0.05},
                    ),
                )
                runner.run()

            trades = pd.read_csv(out_dir / "trades.csv")
            # First trade executes, later trade blocked after drawdown
            self.assertEqual(len(trades), 1)


if __name__ == "__main__":
    unittest.main()
