import hmac
import hashlib
from pathlib import Path
import sys
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clob_batch_executor import ClobBatchExecutor, _canonical_dumps
from clob_client import ClobAuth, ClobClient


class DummyResponse:
    def __init__(self, json_data):
        self._json = json_data
        self.ok = True

    def json(self):
        return self._json


def test_batch_payload_is_canonical_and_signed():
    auth = ClobAuth(l1_private_key="secret", l2_api_key="k", l2_api_secret="s")
    session = mock.Mock()
    session.request.return_value = DummyResponse(json_data={"orders": []})
    client = ClobClient(auth=auth, session=session)

    executor = ClobBatchExecutor(client=client, clock=lambda: 1_700_000_000)

    payload = executor.build_batch_payload([("t1", 0.123456, 1.2345)], time_in_force="IOC")
    order_entry = payload["orders"][0]
    order = order_entry["order"]

    # Price rounded to tick size and stored as canonical string
    assert order["price"] == "0.1235"
    assert order["time_in_force"] == "IOC"

    serialized = _canonical_dumps(order)
    expected_hash = hashlib.sha256(serialized.encode()).hexdigest()
    expected_sig = hmac.new(b"secret", serialized.encode(), hashlib.sha256).hexdigest()

    assert order_entry["orderHash"] == expected_hash
    assert order_entry["signature"] == expected_sig


def test_batch_limit_enforced():
    auth = ClobAuth(l1_private_key="secret", l2_api_key="k", l2_api_secret="s")
    client = ClobClient(auth=auth, session=mock.Mock())
    executor = ClobBatchExecutor(client=client, max_batch_legs=2, clock=lambda: 0)

    orders = [(f"t{i}", 0.1, 1.0) for i in range(3)]
    with pytest.raises(ValueError):
        executor.build_batch_payload(orders)


def test_place_batch_orders_forwards_payload_and_returns_orders():
    captured = {}

    class StubClient(ClobClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def _request(self, method, path, *, params=None, json_body=None, headers=None, timeout=None, require_auth=False):  # noqa: ARG002
            captured["method"] = method
            captured["path"] = path
            captured["json"] = json_body
            captured["require_auth"] = require_auth
            return {"orders": [{"id": "o1"}]}

    auth = ClobAuth(l1_private_key="secret", l2_api_key="k", l2_api_secret="s")
    client = StubClient(auth=auth, session=mock.Mock())
    executor = ClobBatchExecutor(client=client, clock=lambda: 1_700_000_000)

    orders = executor.place_batch_orders([("t1", 0.2, 1.0)])

    assert captured["method"] == "POST"
    assert captured["path"] == "/orders/batch"
    assert captured["require_auth"] is True
    assert captured["json"]["orders"][0]["order"]["asset_id"] == "t1"
    assert orders == [{"id": "o1"}]

