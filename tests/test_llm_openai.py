import json
import os
import unittest
from unittest import mock

from research.llm_openai import ResearchLLMClient
from research.schema import ResearchFeatures
from trading_bot import TradingBotConfig, compute_backoff_ms


def _response(text: str):
    class Content:
        def __init__(self, t: str):
            self.text = t

    class Output:
        def __init__(self, t: str):
            self.content = [Content(t)]

    class Response:
        def __init__(self, t: str):
            self.output = [Output(t)]

    return Response(text)


class FakeResponses:
    def __init__(self, mapping):
        self.mapping = {k: list(v) for k, v in mapping.items()}
        self.calls = []

    def create(self, model=None, **_):
        self.calls.append(model)
        queue = self.mapping.get(model, [])
        if not queue:
            raise AssertionError("Unexpected model call")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, mapping):
        self.responses = FakeResponses(mapping)


class ResearchLLMClientTests(unittest.TestCase):
    def setUp(self):
        self.schema = ResearchFeatures.json_schema()

    def test_valid_schema_returns_primary(self):
        payload = json.dumps({"llm_p_yes": 0.4, "llm_confidence": 0.8})
        client = FakeClient({"gpt-5-mini": [_response(payload)]})
        llm = ResearchLLMClient("gpt-5-mini", "gpt-5.2", api_key="test", client=client)

        data, from_cache = llm.call_llm("m1", 0, "prompt", self.schema)

        self.assertFalse(from_cache)
        self.assertEqual(client.responses.calls, ["gpt-5-mini"])
        self.assertAlmostEqual(data["llm_p_yes"], 0.4)

    def test_low_confidence_triggers_fallback(self):
        primary = json.dumps({"llm_p_yes": 0.1, "llm_confidence": 0.1})
        fallback = json.dumps({"llm_p_yes": 0.7, "llm_confidence": 0.9})
        client = FakeClient({"gpt-5-mini": [_response(primary)], "gpt-5.2": [_response(fallback)]})
        llm = ResearchLLMClient("gpt-5-mini", "gpt-5.2", api_key="test", client=client)

        data, _ = llm.call_llm("m1", 0, "prompt", self.schema)

        self.assertEqual(client.responses.calls, ["gpt-5-mini", "gpt-5.2"])
        self.assertAlmostEqual(data["llm_p_yes"], 0.7)

    def test_schema_violation_triggers_fallback(self):
        client = FakeClient({"gpt-5-mini": [_response("not json")], "gpt-5.2": [_response(json.dumps({"llm_confidence": 0.9}))]})
        llm = ResearchLLMClient("gpt-5-mini", "gpt-5.2", api_key="test", client=client)

        data, _ = llm.call_llm("m1", 0, "prompt", self.schema)

        self.assertEqual(client.responses.calls, ["gpt-5-mini", "gpt-5.2"])
        self.assertAlmostEqual(data["llm_confidence"], 0.9)

    def test_trading_bot_path_does_not_require_openai_key(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True):
            cfg = TradingBotConfig()
            backoff = compute_backoff_ms(0, cfg.api_backoff_base_ms, cfg.api_backoff_max_ms)
            self.assertGreaterEqual(backoff, 0)


if __name__ == "__main__":
    unittest.main()
