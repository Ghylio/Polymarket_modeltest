import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from data.build_snapshots import HAS_PARQUET, SnapshotBuilder


class SnapshotBuilderTests(unittest.TestCase):
    def setUp(self):
        self.builder = SnapshotBuilder(fetcher=None, use_sentiment=False)

    def _sample_market(self, outcome="YES", market_id="m1"):
        resolution_time = datetime(2024, 1, 10, 12, 0, 0)
        return {
            "id": market_id,
            "eventId": f"event-{market_id}",
            "resolutionTime": resolution_time.isoformat(),
            "endDate": resolution_time.isoformat(),
            "winningOutcome": outcome,
        }

    def _sample_trades(self, resolution_time: datetime):
        timestamps = [
            resolution_time - timedelta(days=5),
            resolution_time - timedelta(days=3),
            resolution_time - timedelta(days=1),
            resolution_time - timedelta(hours=5),
            resolution_time - timedelta(minutes=30),
        ]
        prices = [0.45, 0.48, 0.52, 0.58, 0.62]
        bids = [0.44, 0.47, 0.51, 0.57, 0.61]
        asks = [0.46, 0.49, 0.53, 0.59, 0.63]
        sides = ["buy", "sell", "buy", "sell", "buy"]

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "price": prices,
                "size": [100, 120, 80, 90, 110],
                "side": sides,
                "bid": bids,
                "ask": asks,
            }
        )

    def test_build_snapshots_generates_buckets(self):
        market = self._sample_market(outcome="YES", market_id="m1")
        trades = self._sample_trades(datetime.fromisoformat(market["resolutionTime"]))

        dataset = self.builder.build_snapshots([market], {"m1": trades})

        expected_buckets = {"3d", "1d", "6h", "1h"}  # feasible given trade coverage
        self.assertEqual(set(dataset["time_bucket"]), expected_buckets)
        self.assertTrue((dataset["y"] == 1).all())
        # Ensure snapshot timestamps are before or equal resolution
        resolution_ts = pd.to_datetime(market["resolutionTime"])
        self.assertTrue((dataset["snapshot_ts"] <= resolution_ts).all())
        # Ensure mid prices come from truncated trades (no leakage)
        latest_mid = dataset.loc[dataset["time_bucket"] == "1h", "p_mid"].iloc[0]
        self.assertAlmostEqual(latest_mid, 0.58, places=4)

    def test_labels_handle_no_outcome(self):
        market = self._sample_market(outcome="NO", market_id="m2")
        trades = self._sample_trades(datetime.fromisoformat(market["resolutionTime"]))

        dataset = self.builder.build_snapshots([market], {"m2": trades})

        self.assertTrue((dataset["y"] == 0).all())

    @unittest.skipUnless(HAS_PARQUET, "pyarrow is required for parquet persistence")
    def test_save_to_parquet(self):
        market = self._sample_market(outcome="YES", market_id="m3")
        trades = self._sample_trades(datetime.fromisoformat(market["resolutionTime"]))
        dataset = self.builder.build_snapshots([market], {"m3": trades})

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = self.builder.save(dataset, Path(tmpdir) / "snapshots.parquet")
            self.assertTrue(out_path.exists())
            loaded = pd.read_parquet(out_path)
            self.assertEqual(len(loaded), len(dataset))


if __name__ == "__main__":
    unittest.main()
