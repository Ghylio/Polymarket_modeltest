import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from prediction_model import PolymarketPredictor
from data.snapshot_dataset import SnapshotDatasetLoader

try:
    import pyarrow  # noqa: F401

    HAS_PARQUET = True
except ImportError:  # pragma: no cover - environment without parquet engine
    HAS_PARQUET = False


def _snapshot_parquet(tmpdir: Path) -> Path:
    base_time = datetime(2024, 1, 15)
    rows = []
    for idx in range(12):
        market_id = f"m{idx // 4}"
        rows.append(
            {
                "market_id": market_id,
                "snapshot_ts": base_time - timedelta(hours=idx),
                "time_to_resolve_hours": float(idx + 1),
                "p_mid": 0.5 + 0.02 * (idx % 2),
                "spread": 0.01,
                "feat_a": float(idx),
                "y": int(idx % 2 == 0),
            }
        )
    df = pd.DataFrame(rows)
    path = tmpdir / "snap.parquet"
    df.to_parquet(path)
    return path


def _snapshot_parquet_with_sentiment(tmpdir: Path) -> Path:
    base_time = datetime(2024, 2, 1)
    rows = []
    for idx in range(10):
        market_id = f"s{idx // 5}"
        rows.append(
            {
                "market_id": market_id,
                "snapshot_ts": base_time - timedelta(hours=idx),
                "time_to_resolve_hours": float(idx + 1),
                "p_mid": 0.4 + 0.01 * idx,
                "spread": 0.02,
                "feat_a": float(idx),
                "sent_mean_1h": 0.1 * idx,
                "sent_std_1h": 0.01 * idx,
                "doc_count_1h": float(idx),
                "sent_mean_24h": 0.05 * idx,
                "sent_std_24h": 0.02 * idx,
                "doc_count_24h": float(idx) + 1,
                "sent_mean_6h": 0.02 * idx,
                "sent_std_6h": 0.01 * idx,
                "doc_count_6h": float(idx) + 2,
                "sent_mean_7d": 0.03 * idx,
                "sent_std_7d": 0.01 * idx,
                "doc_count_7d": float(idx) + 3,
                "sent_trend": 0.05,
                "y": int(idx % 2 == 0),
            }
        )
    df = pd.DataFrame(rows)
    path = tmpdir / "snap_sent.parquet"
    df.to_parquet(path)
    return path


class SnapshotTrainingTests(unittest.TestCase):
    @unittest.skipUnless(HAS_PARQUET, "pyarrow is required for parquet snapshot tests")
    def test_predictor_trains_on_snapshots_and_saves_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            parquet_path = _snapshot_parquet(tmp_path)

            loader = SnapshotDatasetLoader()
            dataset = loader.load(parquet_path, val_fraction=0.25)

            predictor = PolymarketPredictor()
            predictor.train(
                dataset.X_train,
                dataset.y_train,
                feature_columns=dataset.feature_columns,
                metadata={"dataset_path": str(dataset.dataset_path)},
            )

            out_path = tmp_path / "model.joblib"
            predictor.save_artifact(str(out_path))
            self.assertTrue(out_path.exists())

            loaded = PolymarketPredictor.load_artifact(str(out_path))
            self.assertEqual(loaded.training_metadata.get("mode"), "resolution")
            self.assertEqual(loaded.training_metadata.get("dataset_path"), str(parquet_path))
            self.assertEqual(loaded.feature_columns, dataset.feature_columns)
            self.assertIsNotNone(loaded.training_metrics.get("baseline_brier"))
            self.assertIsNotNone(loaded.training_metrics.get("baseline_logloss"))
            self.assertTrue(loaded.training_metrics.get("calibration_table"))

    def test_directional_mode_requires_explicit_flag(self):
        predictor = PolymarketPredictor()
        training_sample = []
        for i in range(12):
            training_sample.append({
                'features': np.zeros(10),
                'future_price': 0.5,
                'outcome': i % 2,
                'current_price': 0.5,
            })

        with mock.patch.object(predictor, '_train_directional', return_value={'status': 'legacy_called'}) as legacy_train:
            with self.assertRaises(ValueError):
                predictor.train(training_sample)
            legacy_train.assert_not_called()

            predictor.train(training_sample, legacy_directional=True)
            legacy_train.assert_called_once_with(training_sample)

    def test_inference_fills_missing_sentiment_with_nans(self):
        feature_cols = [
            "p_mid",
            "feat_a",
            "sent_mean_1h",
            "sent_std_1h",
            "doc_count_1h",
            "sent_trend",
        ]
        X = np.random.rand(20, len(feature_cols))
        y = np.random.randint(0, 2, size=20)

        predictor = PolymarketPredictor()
        predictor.train(
            X,
            y,
            feature_columns=feature_cols,
            metadata={
                "dataset_path": "synthetic",
                "sentiment_enabled_at_train": True,
                "sentiment_feature_columns_used": [c for c in feature_cols if "sent_" in c],
            },
        )

        predictor.sentiment_enabled = False  # Force NaN fill path
        market = {"id": "m1", "outcomePrices": [0.5, 0.5]}
        trades_df = pd.DataFrame({"price": [0.5, 0.51], "size": [1, 1]})

        result = predictor.predict_probability(market, trades_df)
        self.assertIn("p_mean", result)
        self.assertEqual(result["features"].shape[1], len(feature_cols))

    @unittest.skipUnless(HAS_PARQUET, "pyarrow is required for parquet snapshot tests")
    def test_sentiment_columns_used_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            parquet_path = _snapshot_parquet_with_sentiment(tmp_path)

            loader = SnapshotDatasetLoader()
            dataset = loader.load(parquet_path, val_fraction=0.25, use_sentiment=True)
            self.assertTrue(dataset.sentiment_feature_columns)

            predictor = PolymarketPredictor()
            predictor.train(
                dataset.X_train,
                dataset.y_train,
                feature_columns=dataset.feature_columns,
                metadata={
                    "dataset_path": str(dataset.dataset_path),
                    "sentiment_enabled_at_train": dataset.sentiment_enabled,
                    "sentiment_feature_columns_used": dataset.sentiment_feature_columns,
                },
            )

            self.assertTrue(predictor.training_metadata.get("sentiment_enabled_at_train"))
            self.assertListEqual(
                predictor.training_metadata.get("sentiment_feature_columns_used", []),
                dataset.sentiment_feature_columns,
            )


if __name__ == "__main__":
    unittest.main()
