import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from data.snapshot_dataset import SnapshotDatasetLoader

try:
    import pyarrow  # noqa: F401

    HAS_PARQUET = True
except ImportError:  # pragma: no cover - environment without parquet engine
    HAS_PARQUET = False


def _build_sample_df():
    base_time = datetime(2024, 1, 10)
    rows = []
    for m_id, label in [("m1", 1), ("m2", 0)]:
        for offset_hours in [120, 48, 6]:
            rows.append(
                {
                    "market_id": m_id,
                    "event_id": f"e-{m_id}",
                    "snapshot_ts": base_time - timedelta(hours=offset_hours),
                    "time_to_resolve_hours": float(offset_hours),
                    "p_mid": 0.55,
                    "spread": 0.02,
                    "feat_a": 0.1 * offset_hours,
                    "feat_b": 0.05,
                    "y": label,
                }
            )
    return pd.DataFrame(rows)


class SnapshotDatasetLoaderTests(unittest.TestCase):
    @unittest.skipUnless(HAS_PARQUET, "pyarrow is required for parquet snapshot tests")
    def test_loader_splits_by_market_no_leakage(self):
        df = _build_sample_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snap.parquet"
            df.to_parquet(path)

            loader = SnapshotDatasetLoader()
            dataset = loader.load(path, val_fraction=0.5)

            self.assertTrue(set(dataset.train_metadata["market_id"]).isdisjoint(
                set(dataset.val_metadata["market_id"])
            ))
            self.assertIn("feat_a", dataset.feature_columns)
            self.assertEqual(
                dataset.X_train.shape[0] + dataset.X_val.shape[0], len(df)
            )

    @unittest.skipUnless(HAS_PARQUET, "pyarrow is required for parquet snapshot tests")
    def test_loader_requires_columns(self):
        df = _build_sample_df().drop(columns=["spread"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snap.parquet"
            df.to_parquet(path)

            loader = SnapshotDatasetLoader()
            with self.assertRaises(ValueError):
                loader.load(path)


if __name__ == "__main__":
    unittest.main()
