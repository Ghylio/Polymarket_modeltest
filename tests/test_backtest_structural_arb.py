import unittest

import pandas as pd

from backtest.run import BacktestConfig, replay_structural_arb


class TestStructuralArbReplay(unittest.TestCase):
    def test_replay_records_edge_and_win(self):
        snapshots = pd.DataFrame(
            [
                {
                    "snapshot_ts": pd.Timestamp("2024-01-01 00:00:00"),
                    "event_id": "event-1",
                    "token_id": "t1",
                    "ask": 0.4,
                    "depth": 1.5,
                },
                {
                    "snapshot_ts": pd.Timestamp("2024-01-01 00:00:00"),
                    "event_id": "event-1",
                    "token_id": "t2",
                    "ask": 0.5,
                    "depth": 1.5,
                },
            ]
        )
        config = BacktestConfig(
            arb_enabled=True,
            arb_edge_min=0.0,
            arb_slippage_buffer=0.0,
            arb_execution_buffer=0.0,
            arb_size=1.0,
            slippage_bps=0.0,
        )

        state = replay_structural_arb(snapshots, config)

        self.assertEqual(state.attempts, 1)
        self.assertEqual(state.wins, 1)
        self.assertEqual(state.losses, 0)
        self.assertAlmostEqual(state.pnl, 0.1)

    def test_partial_fill_triggers_cooldown(self):
        snapshots = pd.DataFrame(
            [
                {
                    "snapshot_ts": pd.Timestamp("2024-01-01 00:00:00"),
                    "event_id": "event-1",
                    "token_id": "t1",
                    "ask": 0.45,
                    "depth": 0.5,
                },
                {
                    "snapshot_ts": pd.Timestamp("2024-01-01 00:00:00"),
                    "event_id": "event-1",
                    "token_id": "t2",
                    "ask": 0.45,
                    "depth": 2.0,
                },
                {
                    "snapshot_ts": pd.Timestamp("2024-01-01 00:05:00"),
                    "event_id": "event-1",
                    "token_id": "t1",
                    "ask": 0.45,
                    "depth": 2.0,
                },
                {
                    "snapshot_ts": pd.Timestamp("2024-01-01 00:05:00"),
                    "event_id": "event-1",
                    "token_id": "t2",
                    "ask": 0.45,
                    "depth": 2.0,
                },
            ]
        )
        config = BacktestConfig(
            arb_enabled=True,
            arb_edge_min=0.0,
            arb_slippage_buffer=0.0,
            arb_execution_buffer=0.0,
            arb_size=1.0,
            cooldown_after_partial_fill_sec=400,
        )

        state = replay_structural_arb(snapshots, config)

        self.assertEqual(state.attempts, 1)
        self.assertEqual(state.partial_fills, 1)
        self.assertGreater(state.attempt_loss, 0)

    def test_depth_filter_blocks_group(self):
        snapshots = pd.DataFrame(
            [
                {
                    "snapshot_ts": pd.Timestamp("2024-01-01 00:00:00"),
                    "event_id": "event-2",
                    "token_id": "t1",
                    "ask": 0.55,
                    "depth": 0.25,
                },
                {
                    "snapshot_ts": pd.Timestamp("2024-01-01 00:00:00"),
                    "event_id": "event-2",
                    "token_id": "t2",
                    "ask": 0.4,
                    "depth": 0.25,
                },
            ]
        )
        config = BacktestConfig(
            arb_enabled=True,
            arb_min_depth_per_leg=0.5,
            arb_edge_min=0.0,
            arb_slippage_buffer=0.0,
            arb_execution_buffer=0.0,
        )

        state = replay_structural_arb(snapshots, config)

        self.assertEqual(state.attempts, 0)
        self.assertEqual(state.pnl, 0.0)


if __name__ == "__main__":
    unittest.main()
