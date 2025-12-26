from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.subgraph_client import SubgraphClient
from data.subgraph_features import SubgraphFeatureJoiner


class StubClient(SubgraphClient):
    def __init__(self, trades=None, liquidity=None, resolutions=None):
        super().__init__(subgraph_url="stub", resolution_url="stub")
        self._trades = trades or {}
        self._liquidity = liquidity or {}
        self._resolutions = resolutions or {}

    def fetch_market_trades(self, market_id: str, start_ts=None, end_ts=None):
        return self._trades.get(market_id, pd.DataFrame(columns=["market_id", "price", "amount", "timestamp"])).copy()

    def fetch_liquidity_snapshots(self, market_id: str, start_ts=None, end_ts=None):
        return self._liquidity.get(market_id, pd.DataFrame(columns=["market_id", "liquidity", "timestamp"])).copy()

    def fetch_resolution_events(self, market_ids):
        rows = []
        for market_id in market_ids:
            if market_id in self._resolutions:
                rows.append(self._resolutions[market_id])
        if not rows:
            return pd.DataFrame(columns=["market_id", "resolution_ts", "outcome"])
        return pd.DataFrame(rows)


def test_subgraph_client_schema_mapping():
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeSession:
        def __init__(self, payload):
            self.payload = payload

        def post(self, url, json, timeout):
            return FakeResponse(self.payload)

    payload = {
        "data": {
            "trades": [
                {
                    "market": "m1",
                    "outcome": "YES",
                    "price": "0.45",
                    "amount": "12.5",
                    "timestamp": "1700000000",
                }
            ],
            "snapshots": [
                {"market": "m1", "liquidity": "123.4", "timestamp": "1700001000"}
            ],
            "markets": [
                {
                    "id": "m1",
                    "outcome": "YES",
                    "resolutionTime": "2024-02-01T12:00:00Z",
                }
            ],
        }
    }

    client = SubgraphClient(session=FakeSession(payload))

    trades = client.fetch_market_trades("m1")
    liquidity = client.fetch_liquidity_snapshots("m1")
    resolutions = client.fetch_resolution_events(["m1"])

    assert list(trades.columns) == ["market_id", "outcome", "price", "amount", "timestamp"]
    assert trades.iloc[0]["price"] == 0.45
    assert pd.api.types.is_datetime64_any_dtype(trades["timestamp"])

    assert list(liquidity.columns) == ["market_id", "liquidity", "timestamp"]
    assert liquidity.iloc[0]["liquidity"] == 123.4
    assert pd.api.types.is_datetime64_any_dtype(liquidity["timestamp"])

    assert list(resolutions.columns) == ["market_id", "resolution_ts", "outcome"]
    assert pd.api.types.is_datetime64_any_dtype(resolutions["resolution_ts"])
    assert resolutions.iloc[0]["outcome"] == "YES"


def test_leakage_safe_joins():
    base_time = datetime(2024, 2, 1, 12, 0, 0)
    snapshots = pd.DataFrame(
        {
            "market_id": ["m1", "m1"],
            "snapshot_ts": [base_time, base_time + timedelta(hours=2)],
            "p_mid": [0.5, 0.52],
            "y": [1, 1],
            "time_bucket": ["1d", "1h"],
        }
    )

    trades = pd.DataFrame(
        {
            "market_id": ["m1", "m1"],
            "price": [0.5, 0.55],
            "amount": [10, 20],
            "timestamp": [base_time - timedelta(hours=2), base_time + timedelta(hours=2)],
        }
    )

    liquidity = pd.DataFrame(
        {
            "market_id": ["m1", "m1"],
            "liquidity": [100, 150],
            "timestamp": [base_time - timedelta(hours=3), base_time + timedelta(hours=2)],
        }
    )

    resolution_df = pd.DataFrame(
        {
            "market_id": ["m1"],
            "resolution_ts": [base_time + timedelta(hours=1, minutes=30)],
            "outcome": ["YES"],
        }
    )

    client = StubClient(trades={"m1": trades}, liquidity={"m1": liquidity}, resolutions={"m1": resolution_df.iloc[0].to_dict()})
    joiner = SubgraphFeatureJoiner(client=client, volume_lookback_hours=24, enabled=True)

    enriched = joiner.enrich_snapshots(snapshots)

    first_row = enriched.iloc[0]
    second_row = enriched.iloc[1]

    assert first_row["subgraph_volume_24h"] == 10
    assert second_row["subgraph_volume_24h"] == 30  # includes trades through snapshot

    assert first_row["subgraph_liquidity"] == 100
    assert second_row["subgraph_liquidity"] == 150

    assert pd.isna(first_row["subgraph_resolution_outcome"])
    assert second_row["subgraph_resolution_outcome"] == "YES"
    assert second_row["subgraph_resolution_ts"] == resolution_df.iloc[0]["resolution_ts"]
