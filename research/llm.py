"""LLM client with rate limiting, caching, and retries."""
from __future__ import annotations

import os
import threading
import time
from typing import Dict, Tuple

import requests

from research.prompt import parse_llm_output


class TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.timestamp = time.monotonic()
        self.lock = threading.Lock()

    def consume(self, tokens: float = 1.0) -> bool:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.timestamp
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.timestamp = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


class LLMClient:
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        cache_ttl_sec: int = 3600,
        rate_limit_per_min: int = 60,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.cache_ttl_sec = cache_ttl_sec
        self.cache: Dict[Tuple[str, int], Tuple[float, Dict]] = {}
        self.bucket = TokenBucket(rate=rate_limit_per_min / 60.0, capacity=float(rate_limit_per_min))

    # ------------------------------------------------------------------
    def has_cache(self, market_id: str, hour_bucket: int) -> bool:
        key = (market_id, hour_bucket)
        if key not in self.cache:
            return False
        ts, _ = self.cache[key]
        return (time.time() - ts) < self.cache_ttl_sec

    def call_llm(self, market_id: str, hour_bucket: int, prompt: str) -> Tuple[Dict, bool]:
        """Call the configured LLM, returning parsed JSON and cache flag."""

        key = (market_id, hour_bucket)
        if self.has_cache(market_id, hour_bucket):
            return self.cache[key][1], True

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }

        last_exc: Exception | None = None
        for attempt in range(3):
            if not self.bucket.consume():
                raise RuntimeError("LLM rate limited")
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                raw_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                parsed = parse_llm_output(raw_text)
                parsed_dict = parsed.dict()
                self.cache[key] = (time.time(), parsed_dict)
                return parsed_dict, False
            except Exception as exc:  # pragma: no cover - network path
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(f"LLM request failed after retries: {last_exc}")
