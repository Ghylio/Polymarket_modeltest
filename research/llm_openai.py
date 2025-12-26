"""OpenAI Responses client wrapper with structured outputs and fallback."""
from __future__ import annotations

import os
import time
from typing import Dict, Tuple

try:  # pragma: no cover - dependency shim for environments without openai installed
    from openai import APIStatusError, APITimeoutError, OpenAI, RateLimitError
except ImportError:  # pragma: no cover - fallback when openai is not available during tests
    class _PlaceholderError(Exception):
        pass

    class APIStatusError(_PlaceholderError):
        status_code: int | None = None

    class APITimeoutError(_PlaceholderError):
        pass

    class RateLimitError(_PlaceholderError):
        pass

    class OpenAI:  # type: ignore
        def __init__(self, *_, **__):
            raise ImportError("openai package is required to use ResearchLLMClient")

from research.prompt import parse_llm_output


class TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.timestamp = time.monotonic()

    def consume(self, tokens: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self.timestamp
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.timestamp = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class ResearchLLMClient:
    """Structured-output OpenAI client with primary/fallback models."""

    def __init__(
        self,
        primary_model: str,
        fallback_model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        cache_ttl_sec: int = 3600,
        rate_limit_per_min: int = 60,
        fallback_confidence_threshold: float = 0.55,
        request_timeout_sec: int = 60,
        max_retries: int = 3,
        store: bool = False,
        client: OpenAI | None = None,
        structured_outputs: bool = True,
    ):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = client or OpenAI(
            api_key=self.api_key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            organization=organization or os.getenv("OPENAI_ORG_ID"),
            timeout=request_timeout_sec,
        )
        self.cache_ttl_sec = cache_ttl_sec
        self.cache: Dict[Tuple[str, int], Tuple[float, Dict]] = {}
        self.bucket = TokenBucket(rate=rate_limit_per_min / 60.0, capacity=float(rate_limit_per_min))
        self.fallback_confidence_threshold = fallback_confidence_threshold
        self.max_retries = max_retries
        self.store = store
        self.structured_outputs = structured_outputs

    # ------------------------------------------------------------------
    def has_cache(self, market_id: str, hour_bucket: int) -> bool:
        key = (market_id, hour_bucket)
        if key not in self.cache:
            return False
        ts, _ = self.cache[key]
        return (time.time() - ts) < self.cache_ttl_sec

    # ------------------------------------------------------------------
    def call_llm(self, market_id: str, hour_bucket: int, prompt: str, schema: Dict) -> Tuple[Dict, bool]:
        """Call primary model and fall back if needed. Returns parsed JSON and cache flag."""

        key = (market_id, hour_bucket)
        if self.has_cache(market_id, hour_bucket):
            return self.cache[key][1], True

        primary_result = None
        try:
            primary_result = self._call_with_model(self.primary_model, prompt, schema)
        except Exception:
            primary_result = None

        use_fallback = primary_result is None or self._should_fallback(primary_result)

        if use_fallback:
            fallback_result = self._call_with_model(self.fallback_model, prompt, schema)
            final_result = fallback_result
        else:
            final_result = primary_result

        parsed_dict = final_result.dict()
        self.cache[key] = (time.time(), parsed_dict)
        return parsed_dict, False

    # ------------------------------------------------------------------
    def _call_with_model(self, model_name: str, prompt: str, schema: Dict):
        if not self.bucket.consume():
            raise RuntimeError("LLM rate limited")

        errors: list[Exception] = []
        backoff = 1.0
        for attempt in range(self.max_retries):
            try:
                response_format = None
                if self.structured_outputs:
                    response_format = {
                        "type": "json_schema",
                        "json_schema": {"name": "research_features", "schema": schema, "strict": True},
                    }
                response = self.client.responses.create(
                    model=model_name,
                    input=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                    temperature=0.2,
                    response_format=response_format,
                    store=self.store,
                )
                raw_text = self._extract_text(response)
                return parse_llm_output(raw_text)
            except RateLimitError as exc:
                errors.append(exc)
            except APITimeoutError as exc:
                errors.append(exc)
            except APIStatusError as exc:
                errors.append(exc)
                if exc.status_code is not None and exc.status_code < 500:
                    raise
            except Exception as exc:
                errors.append(exc)
                raise

            if attempt < self.max_retries - 1:
                time.sleep(backoff)
                backoff *= 1.5
        raise RuntimeError(f"LLM request failed after retries: {errors[-1] if errors else 'unknown error'}")

    # ------------------------------------------------------------------
    def _extract_text(self, response) -> str:
        if hasattr(response, "output_text"):
            return response.output_text
        output = getattr(response, "output", None)
        if output:
            first = output[0]
            content = getattr(first, "content", None) or []
            if content:
                first_item = content[0]
                if isinstance(first_item, dict):
                    return first_item.get("text") or first_item.get("input_text") or ""
                if hasattr(first_item, "text"):
                    return getattr(first_item, "text")
        raise ValueError("Could not extract text from OpenAI response")

    def _should_fallback(self, features) -> bool:
        confidence = getattr(features, "llm_confidence", None)
        return confidence is None or (confidence < self.fallback_confidence_threshold)
