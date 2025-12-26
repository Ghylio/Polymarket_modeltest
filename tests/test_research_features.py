import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import unittest

from data.build_snapshots import SnapshotBuilder
from prediction_model import PolymarketPredictor
from research.schema import RESEARCH_COLUMNS, ResearchFeatures
from research.store import ResearchFeatureStore


class DummyExtractor:
    def __init__(self):
        self._feature_names = ["feat_a", "feat_b"]

    def extract_trade_features(self, trades_df):
        return np.array([[0.1, 0.2]])

    def extract_market_features(self, market):
        return np.array([[0.3, 0.4]])

    def combine_features(self, trade_features, market_features):
        return np.array([[0.5, 0.6]])

    def get_feature_names(self):
        return self._feature_names


class ResearchFeatureTests(unittest.TestCase):
    def _builder_with_store(self, store: ResearchFeatureStore) -> SnapshotBuilder:
        builder = SnapshotBuilder(
            buckets=[timedelta(hours=1)],
            use_sentiment=False,
            use_research=True,
            research_store=store,
        )
        builder.feature_extractor = DummyExtractor()
        return builder

    def _basic_market(self):
        return {
            "id": "m1",
            "eventId": "e1",
            "resolutionTime": datetime(2024, 1, 2, 0, 0, 0),
            "outcomePrices": [0.5, 0.5],
            "winningOutcome": "yes",
        }

    def _basic_trades(self):
        return pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 23, 0, 0)],
                "price": [0.5],
                "size": [1.0],
            }
        )

    def test_snapshot_leakage_protection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "research.db"
            store = ResearchFeatureStore(db_path)

            market = self._basic_market()
            resolution_ts = pd.to_datetime(market["resolutionTime"])
            snapshot_ts = int((resolution_ts - timedelta(hours=1)).timestamp())

            store.upsert_research_features(
                market_id="m1",
                as_of_ts=snapshot_ts + 3600,
                features=ResearchFeatures(llm_p_yes=0.9),
                raw_json={"test": True},
            )

            builder = self._builder_with_store(store)
            df = builder.build_snapshots(
                markets=[market], trades_by_market={"m1": self._basic_trades()}
            )

            self.assertIn("llm_p_yes", df.columns)
            self.assertTrue(np.isnan(df.loc[0, "llm_p_yes"]))

    def test_snapshot_defaults_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ResearchFeatureStore(Path(tmpdir) / "research.db")
            builder = self._builder_with_store(store)
            df = builder.build_snapshots(
                markets=[self._basic_market()], trades_by_market={"m1": self._basic_trades()}
            )

            for col in RESEARCH_COLUMNS:
                self.assertIn(col, df.columns)
            self.assertEqual(df.loc[0, "evidence_count_24h"], 0)
            self.assertTrue(np.isnan(df.loc[0, "llm_p_yes"]))
            self.assertTrue(np.isnan(df.loc[0, "ambiguity_score"]))

    def test_live_inference_alignment_with_missing_research(self):
        predictor = PolymarketPredictor()
        feature_cols = ["p_mid"] + RESEARCH_COLUMNS
        predictor.feature_columns = feature_cols
        aligned = predictor._align_feature_frame(pd.DataFrame([{"p_mid": 0.5}]))

        self.assertListEqual(list(aligned.columns), feature_cols)
        self.assertEqual(aligned.loc[0, "evidence_count_24h"], 0)
        self.assertTrue(np.isnan(aligned.loc[0, "llm_p_yes"]))

        result = predictor.predict_probability_from_features(pd.DataFrame([{"p_mid": 0.5}]))
        self.assertIn("p_mean", result)


if __name__ == "__main__":
    unittest.main()
