"""Feature joiner for Polymarket subgraph data."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterable

import pandas as pd

from data.subgraph_client import SubgraphClient


@dataclass
class SubgraphFeatureJoiner:
    """Attach subgraph-derived features to snapshot rows."""

    client: SubgraphClient
    volume_lookback_hours: int = 24
    enabled: bool = True

    def enrich_snapshots(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        """Join volume/liquidity/resolution features into snapshot rows."""

        if not self.enabled or snapshots.empty:
            return snapshots

        enriched = snapshots.copy()
        if "snapshot_ts" not in enriched.columns:
            raise ValueError("snapshot_ts column required for subgraph enrichment")

        enriched["subgraph_volume_24h"] = float("nan")
        enriched["subgraph_liquidity"] = float("nan")
        enriched["subgraph_resolution_outcome"] = None
        enriched["subgraph_resolution_ts"] = pd.NaT

        unique_markets = enriched["market_id"].unique().tolist()
        resolutions = self._resolution_map(unique_markets)

        for market_id in unique_markets:
            market_rows = enriched[enriched["market_id"] == market_id]
            trades = self.client.fetch_market_trades(market_id)
            liquidity = self.client.fetch_liquidity_snapshots(market_id)
            res_meta = resolutions.get(market_id)

            for idx, row in market_rows.iterrows():
                ts = pd.to_datetime(row["snapshot_ts"])
                enriched.loc[idx, "subgraph_volume_24h"] = self._volume_in_window(
                    trades, ts
                )
                enriched.loc[idx, "subgraph_liquidity"] = self._liquidity_at_ts(
                    liquidity, ts
                )

                if res_meta is not None and pd.notna(res_meta["resolution_ts"]):
                    if res_meta["resolution_ts"] <= ts:
                        enriched.loc[idx, "subgraph_resolution_ts"] = res_meta[
                            "resolution_ts"
                        ]
                        enriched.loc[idx, "subgraph_resolution_outcome"] = res_meta[
                            "outcome"
                        ]

        return enriched

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _volume_in_window(self, trades: pd.DataFrame, ts: pd.Timestamp) -> float:
        if trades.empty:
            return float("nan")
        window_start = ts - timedelta(hours=self.volume_lookback_hours)
        mask = (trades["timestamp"] > window_start) & (trades["timestamp"] <= ts)
        return trades.loc[mask, "amount"].sum()

    @staticmethod
    def _liquidity_at_ts(liquidity: pd.DataFrame, ts: pd.Timestamp) -> float:
        if liquidity.empty:
            return float("nan")
        eligible = liquidity[liquidity["timestamp"] <= ts]
        if eligible.empty:
            return float("nan")
        return eligible.iloc[-1]["liquidity"]

    def _resolution_map(self, market_ids: Iterable[str]) -> Dict[str, Dict]:
        resolution_df = self.client.fetch_resolution_events(market_ids)
        resolution_map: Dict[str, Dict] = {}
        for _, row in resolution_df.iterrows():
            resolution_map[str(row["market_id"])] = {
                "resolution_ts": row["resolution_ts"],
                "outcome": row.get("outcome"),
            }
        return resolution_map
