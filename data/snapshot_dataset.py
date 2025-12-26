"""Snapshot dataset loader for resolution-based training."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from data.sentiment_features import SENTIMENT_COLUMNS, SENTIMENT_ALIAS_MAP
from research.schema import RESEARCH_COLUMNS


REQUIRED_COLUMNS = {
    "market_id",
    "snapshot_ts",
    "time_to_resolve_hours",
    "y",
    "p_mid",
    "spread",
}

LOGGER = logging.getLogger(__name__)


@dataclass
class SnapshotDataset:
    feature_columns: List[str]
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    train_metadata: pd.DataFrame
    val_metadata: pd.DataFrame
    dataset_path: Path
    sentiment_feature_columns: List[str]
    sentiment_enabled: bool
    research_feature_columns: List[str]
    research_enabled: bool


class SnapshotDatasetLoader:
    """Load resolved snapshot datasets for model training."""

    def __init__(
        self,
        label_col: str = "y",
        time_col: str = "snapshot_ts",
        market_col: str = "market_id",
    ):
        self.label_col = label_col
        self.time_col = time_col
        self.market_col = market_col

    def load(
        self,
        path: Path,
        val_fraction: float = 0.2,
        required_extra: Optional[Iterable[str]] = None,
        use_sentiment: bool = True,
        use_research: bool = True,
    ) -> SnapshotDataset:
        """Load parquet snapshots and return train/val splits without leakage."""

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Snapshot dataset not found at {path}")

        df = pd.read_parquet(path)
        df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
        df = df.dropna(subset=[self.time_col]).reset_index(drop=True)

        # Backward-compatible alias mapping for sentiment columns
        for alias, target in SENTIMENT_ALIAS_MAP.items():
            for prefix in ("sent_mean_", "sent_std_", "doc_count_"):
                alias_col = f"{prefix}{alias}"
                target_col = f"{prefix}{target}"
                if alias_col in df.columns and target_col not in df.columns:
                    df[target_col] = df[alias_col]

        required = set(REQUIRED_COLUMNS)
        if required_extra:
            required.update(required_extra)

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Snapshot dataset missing required columns: {sorted(missing)}")

        if self.label_col not in df.columns:
            raise ValueError(f"Label column '{self.label_col}' missing from dataset")

        # Avoid leakage by assigning entire markets to either train or validation
        market_last_ts = (
            df.groupby(self.market_col)[self.time_col]
            .max()
            .sort_values()
        )
        n_markets = len(market_last_ts)
        if n_markets == 0:
            raise ValueError("Snapshot dataset is empty after preprocessing")

        cutoff = int(max(n_markets * (1 - val_fraction), 1))
        train_markets = set(market_last_ts.iloc[:cutoff].index)
        val_markets = set(market_last_ts.iloc[cutoff:].index)
        if not val_markets:
            # Ensure at least one market in validation if dataset small
            val_markets = {market_last_ts.index[-1]}
            train_markets = set(market_last_ts.index[:-1]) or val_markets

        meta_cols = [self.market_col, self.time_col]
        exclude_cols = set(meta_cols + [self.label_col, "schema_version", "time_bucket"])

        sentiment_cols_present = [c for c in SENTIMENT_COLUMNS if c in df.columns]
        if not use_sentiment:
            feature_columns = [
                c for c in df.columns if c not in exclude_cols and c not in SENTIMENT_COLUMNS
            ]
            sentiment_enabled = False
        else:
            feature_columns = [c for c in df.columns if c not in exclude_cols]
            sentiment_enabled = bool(sentiment_cols_present)
            if not sentiment_cols_present:
                LOGGER.info(
                    "Sentiment requested but columns missing; proceeding without sentiment."
                )
                feature_columns = [
                    c for c in feature_columns if c not in SENTIMENT_COLUMNS
                ]

        sentiment_feature_columns = [c for c in feature_columns if c in SENTIMENT_COLUMNS]

        if not use_research:
            feature_columns = [c for c in feature_columns if c not in RESEARCH_COLUMNS]
        research_feature_columns = [c for c in feature_columns if c in RESEARCH_COLUMNS]
        research_enabled = bool(research_feature_columns)

        train_df = df[df[self.market_col].isin(train_markets)].copy()
        val_df = df[df[self.market_col].isin(val_markets)].copy()

        X_train = train_df[feature_columns]
        y_train = train_df[self.label_col].astype(float)
        X_val = val_df[feature_columns]
        y_val = val_df[self.label_col].astype(float)

        train_meta = train_df[meta_cols]
        val_meta = val_df[meta_cols]

        return SnapshotDataset(
            feature_columns=feature_columns,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            train_metadata=train_meta,
            val_metadata=val_meta,
            dataset_path=path,
            sentiment_feature_columns=sentiment_feature_columns,
            sentiment_enabled=sentiment_enabled,
            research_feature_columns=research_feature_columns,
            research_enabled=research_enabled,
        )

