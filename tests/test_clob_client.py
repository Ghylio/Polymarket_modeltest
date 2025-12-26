from pathlib import Path
from unittest import mock

import pytest

# Ensure repository root is importable for local module resolution
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clob_client import ClobAuth, ClobClient, ClobRequestError


_NO_JSON = object()


class DummyResponse:
    def __init__(self, ok=True, status_code=200, json_data=_NO_JSON, text=""):
        self.ok = ok
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        if self._json is _NO_JSON:
            raise ValueError("invalid json")
        return self._json


def test_build_order_payload_shapes_and_rounding():
    auth = ClobAuth(l1_private_key="l1", l2_api_key="k", l2_api_secret="s")
    session = mock.Mock()
    session.request.return_value = DummyResponse(json_data={"orderId": "abc"})

    client = ClobClient(auth=auth, session=session)
    response = client.place_order("token", "buy", size=1.2, price=0.123456, client_order_id="client-1")

    assert response == {"orderId": "abc"}
    args, kwargs = session.request.call_args
    assert kwargs["json"]["price"] == 0.1235  # tick rounding to 1e-4
    assert kwargs["json"]["side"] == "BUY"
    assert kwargs["json"]["signature"]  # signature always present


def test_authenticated_headers_added_for_private_calls():
    auth = ClobAuth(l1_private_key="l1", l2_api_key="api-key", l2_api_secret="secret")
    session = mock.Mock()
    session.request.return_value = DummyResponse(json_data={"orderId": "abc"})

    client = ClobClient(auth=auth, session=session)
    client.place_order("token", "sell", size=1.0, price=0.25)

    _, kwargs = session.request.call_args
    headers = kwargs["headers"]
    assert headers["X-API-KEY"] == "api-key"
    assert headers.get("X-SIGNATURE")
    assert headers.get("X-TIMESTAMP")


def test_public_market_data_requests_are_unauthenticated():
    auth = ClobAuth(l1_private_key="l1", l2_api_key="k", l2_api_secret="s")
    session = mock.Mock()
    session.request.return_value = DummyResponse(json_data={"bids": [], "asks": []})

    client = ClobClient(auth=auth, session=session)
    client.get_orderbook("token")

    _, kwargs = session.request.call_args
    assert kwargs["headers"] == {}


def test_error_handling_wraps_response_text():
    auth = ClobAuth(l1_private_key="l1", l2_api_key="k", l2_api_secret="s")
    session = mock.Mock()
    session.request.return_value = DummyResponse(ok=False, status_code=400, text="bad request")

    client = ClobClient(auth=auth, session=session)
    with pytest.raises(ClobRequestError) as excinfo:
        client.get_price("token")

    assert "bad request" in str(excinfo.value)
    assert excinfo.value.status_code == 400
