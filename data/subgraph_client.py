"""Thin GraphQL client for Polymarket subgraphs."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

DEFAULT_SUBGRAPH_URL = "https://api.thegraph.com/subgraphs/name/polymarket/matic-markets"
DEFAULT_RESOLUTION_URL = (
    "https://api.thegraph.com/subgraphs/name/polymarket/resolutions"
)

LOGGER = logging.getLogger(__name__)


@dataclass
class SubgraphClient:
    """GraphQL subgraph client for trades, liquidity, and resolutions."""

    subgraph_url: str = DEFAULT_SUBGRAPH_URL
    resolution_url: str = DEFAULT_RESOLUTION_URL
    session: Optional[requests.Session] = None

    def __post_init__(self):
        self.session = self.session or requests.Session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_market_trades(
        self,
        market_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch market trades from the subgraph."""

        query = """
        query MarketTrades($marketId: String!, $start: BigInt, $end: BigInt) {
          trades: marketTrades(
            where: {market: $marketId, timestamp_gte: $start, timestamp_lte: $end}
            orderBy: timestamp
            orderDirection: asc
            first: 1000
          ) {
            market
            outcome
            price
            amount
            timestamp
          }
        }
        """

        variables = {
            "marketId": market_id,
            "start": start_ts,
            "end": end_ts,
        }
        data = self._execute(self.subgraph_url, query, variables).get("trades") or []
        return self._normalize_trades(data)

    def fetch_liquidity_snapshots(
        self,
        market_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch liquidity snapshots from the subgraph."""

        query = """
        query LiquiditySnapshots($marketId: String!, $start: BigInt, $end: BigInt) {
          snapshots: liquiditySnapshots(
            where: {market: $marketId, timestamp_gte: $start, timestamp_lte: $end}
            orderBy: timestamp
            orderDirection: asc
            first: 1000
          ) {
            market
            liquidity
            timestamp
          }
        }
        """

        variables = {
            "marketId": market_id,
            "start": start_ts,
            "end": end_ts,
        }
        data = self._execute(self.subgraph_url, query, variables).get("snapshots") or []
        return self._normalize_liquidity(data)

    def fetch_resolution_events(self, market_ids: Iterable[str]) -> pd.DataFrame:
        """Fetch resolution metadata for a set of markets."""

        market_list = list(market_ids)
        if not market_list:
            return pd.DataFrame(columns=["market_id", "resolution_ts", "outcome"])

        query = """
        query Resolutions($marketIds: [String!]) {
          markets(where: {id_in: $marketIds}) {
            id
            outcome
            resolutionTime
          }
        }
        """

        variables = {"marketIds": market_list}
        data = self._execute(self.resolution_url, query, variables).get("markets") or []
        return self._normalize_resolutions(data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _execute(self, url: str, query: str, variables: Dict) -> Dict:
        response = self.session.post(
            url,
            json={"query": query, "variables": variables},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        if "errors" in payload:
            LOGGER.warning("Subgraph query errors: %s", payload.get("errors"))
        return payload.get("data") or {}

    @staticmethod
    def _normalize_trades(data: List[Dict]) -> pd.DataFrame:
        trades = pd.DataFrame(data)
        if trades.empty:
            return pd.DataFrame(columns=["market_id", "outcome", "price", "amount", "timestamp"])

        trades = trades.rename(columns={"market": "market_id"})
        trades["price"] = trades["price"].astype(float)
        trades["amount"] = trades["amount"].astype(float)
        trades["timestamp"] = pd.to_datetime(
            pd.to_numeric(trades["timestamp"], errors="coerce"), unit="s"
        )
        return trades[["market_id", "outcome", "price", "amount", "timestamp"]]

    @staticmethod
    def _normalize_liquidity(data: List[Dict]) -> pd.DataFrame:
        snapshots = pd.DataFrame(data)
        if snapshots.empty:
            return pd.DataFrame(columns=["market_id", "liquidity", "timestamp"])

        snapshots = snapshots.rename(columns={"market": "market_id"})
        snapshots["liquidity"] = snapshots["liquidity"].astype(float)
        snapshots["timestamp"] = pd.to_datetime(
            pd.to_numeric(snapshots["timestamp"], errors="coerce"), unit="s"
        )
        return snapshots[["market_id", "liquidity", "timestamp"]]

    @staticmethod
    def _normalize_resolutions(data: List[Dict]) -> pd.DataFrame:
        resolutions = pd.DataFrame(data)
        if resolutions.empty:
            return pd.DataFrame(columns=["market_id", "resolution_ts", "outcome"])

        resolutions = resolutions.rename(columns={"id": "market_id"})
        if "resolutionTime" in resolutions.columns:
            resolutions["resolution_ts"] = pd.to_datetime(
                resolutions["resolutionTime"], errors="coerce"
            )
        else:
            resolutions["resolution_ts"] = pd.NaT
        return resolutions[["market_id", "resolution_ts", "outcome"]]
