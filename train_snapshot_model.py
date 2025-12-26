"""CLI for training resolution probability model from snapshot dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from prediction_model import PolymarketPredictor
from data.snapshot_dataset import SnapshotDatasetLoader
from metrics import MetricsLogger, create_run_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train probability model on snapshot dataset")
    parser.add_argument(
        "--snapshots_path",
        default="data/features/snapshots.parquet",
        help="Path to snapshots parquet (default: data/features/snapshots.parquet)",
    )
    parser.add_argument(
        "--out",
        default="models/snapshot_model.joblib",
        help="Output path for trained model artifact",
    )
    parser.add_argument(
        "--legacy-directional",
        action="store_true",
        help="Use legacy price-direction training instead of resolution labels",
    )
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Optional run directory for metrics (default: auto-create under results/)",
    )
    args = parser.parse_args(argv)

    run_dir: Optional[Path] = (
        Path(args.run_dir) if args.run_dir else create_run_dir()
    )
    metrics_logger = MetricsLogger(run_dir)

    predictor = PolymarketPredictor()

    if not args.legacy_directional:
        loader = SnapshotDatasetLoader()
        dataset = loader.load(Path(args.snapshots_path))
        predictor.train(
            dataset.X_train,
            dataset.y_train,
            feature_columns=dataset.feature_columns,
            metadata={
                "dataset_path": str(dataset.dataset_path),
                "sentiment_enabled_at_train": dataset.sentiment_enabled,
                "sentiment_feature_columns_used": dataset.sentiment_feature_columns,
                "research_enabled_at_train": dataset.research_enabled,
                "research_feature_columns_used": dataset.research_feature_columns,
            },
            metrics_logger=metrics_logger,
        )
    else:
        print("⚠️ Legacy directional training requested; fetching real-time training data.")
        training_data = predictor.fetch_real_training_data()
        predictor.train(training_data, legacy_directional=True, metrics_logger=metrics_logger)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save_artifact(str(out_path))
    print(f"✅ Trained model saved to {out_path}\n📈 Metrics logged to {metrics_logger.path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
