from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from data.sentiment_providers import TwitterProvider


def _provider(tmp_path: Path, **config) -> TwitterProvider:
    return TwitterProvider(
        bearer_token="token",
        enabled=True,
        config=config,
        quota_db_path=tmp_path / "quota.db",
    )


def _fixed_times():
    base = datetime(2024, 1, 2, 3, 4, 5)
    return int(base.timestamp())


def test_quota_exhausted_skips_api(tmp_path):
    provider = _provider(tmp_path, monthly_read_budget=1)
    now_ts = _fixed_times()
    month_key = provider._month_key(now_ts)
    provider.quota_store.consume_quota(month_key, budget=1, amount=1)

    with mock.patch("data.sentiment_providers.time.time", return_value=now_ts), mock.patch(
        "data.sentiment_providers.requests.get"
    ) as mock_get:
        docs = provider.fetch("query", datetime.utcnow() - timedelta(hours=1), datetime.utcnow(), market_id="m1")
        assert docs == []
        mock_get.assert_not_called()


def test_cache_hit_returns_data_without_api(tmp_path):
    provider = _provider(tmp_path)
    now_ts = _fixed_times()
    prepared_query = provider._prepare_query("query")
    cache_key = provider._cache_key(prepared_query, now_ts)
    cached_response = {"data": [{"text": "cached", "created_at": "2024-01-02T03:04:05Z"}]}
    provider.quota_store.set_cache(cache_key, cached_response, now_ts=now_ts)

    with mock.patch("data.sentiment_providers.time.time", return_value=now_ts), mock.patch(
        "data.sentiment_providers.requests.get"
    ) as mock_get:
        docs = provider.fetch("query", datetime.utcnow() - timedelta(hours=1), datetime.utcnow(), market_id="m1")
        assert len(docs) == 1
        assert docs[0]["text"] == "cached"
        mock_get.assert_not_called()


def test_day_cap_blocks_calls(tmp_path):
    provider = _provider(tmp_path, max_markets_per_day=1)
    now_ts = _fixed_times()
    day_key = provider._day_key(now_ts)
    provider.quota_store.increment_daily_count(day_key)

    with mock.patch("data.sentiment_providers.time.time", return_value=now_ts), mock.patch(
        "data.sentiment_providers.requests.get"
    ) as mock_get:
        docs = provider.fetch("query", datetime.utcnow() - timedelta(hours=1), datetime.utcnow(), market_id="m1")
        assert docs == []
        mock_get.assert_not_called()


def test_cooldown_enforced(tmp_path):
    provider = _provider(tmp_path, cooldown_hours_per_market=24)
    now_ts = _fixed_times()
    provider.quota_store.update_market_cooldown("m1", now_ts=now_ts - 3600)

    with mock.patch("data.sentiment_providers.time.time", return_value=now_ts), mock.patch(
        "data.sentiment_providers.requests.get"
    ) as mock_get:
        docs = provider.fetch("query", datetime.utcnow() - timedelta(hours=1), datetime.utcnow(), market_id="m1")
        assert docs == []
        mock_get.assert_not_called()


def test_sufficient_other_docs_skip(tmp_path):
    provider = _provider(tmp_path, min_other_docs_24h=2)
    now_ts = _fixed_times()

    with mock.patch("data.sentiment_providers.time.time", return_value=now_ts), mock.patch(
        "data.sentiment_providers.requests.get"
    ) as mock_get:
        docs = provider.fetch(
            "query",
            datetime.utcnow() - timedelta(hours=1),
            datetime.utcnow(),
            market_id="m1",
            other_docs_24h_count=5,
        )
        assert docs == []
        mock_get.assert_not_called()


def test_normal_path_calls_api_and_updates_state(tmp_path):
    provider = _provider(tmp_path)
    now_ts = _fixed_times()
    api_response = {"data": [{"text": "fresh", "created_at": "2024-01-02T03:04:05Z"}]}

    mock_resp = mock.Mock()
    mock_resp.json.return_value = api_response
    mock_resp.raise_for_status.return_value = None

    with mock.patch("data.sentiment_providers.time.time", return_value=now_ts), mock.patch(
        "data.sentiment_providers.requests.get", return_value=mock_resp
    ) as mock_get:
        docs = provider.fetch("query terms", datetime.utcnow() - timedelta(hours=1), datetime.utcnow(), market_id="m1")

    assert len(docs) == 1
    assert docs[0]["text"] == "fresh"
    mock_get.assert_called_once()

    month_key = provider._month_key(now_ts)
    used, _ = provider.quota_store.get_quota_state(month_key, budget=provider.config["monthly_read_budget"])
    assert used == 1
    assert provider.quota_store.get_daily_count(provider._day_key(now_ts)) == 1
    assert provider.quota_store.get_market_cooldown("m1") == now_ts
    prepared = provider._prepare_query("query terms")
    assert provider.quota_store.get_cache(provider._cache_key(prepared, now_ts), ttl_hours=24, now_ts=now_ts)
