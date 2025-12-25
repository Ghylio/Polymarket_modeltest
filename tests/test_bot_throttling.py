import random
import time

from trading_bot import (
    LiveMarketState,
    ProbabilityTradingBot,
    TokenBucketLimiter,
    TradingBotConfig,
    compute_backoff_ms,
)


def test_per_market_gate_tick_change_or_time():
    config = TradingBotConfig(min_ticks_change_to_recalc=0.01, max_evals_per_market_per_sec=1.0)
    bot = ProbabilityTradingBot(fetcher=object(), bot_config=config)
    market_id = "m1"
    now = time.time()
    bot.last_eval[market_id] = {"time": now, "bid": 0.50, "ask": 0.52}

    # Significant tick move triggers evaluation
    state = LiveMarketState(bid=0.52, ask=0.55, mid=0.535, spread=0.03, depth=10, last_update=now)
    assert bot._should_evaluate(market_id, state)

    # No move but time gate should allow after 1s
    bot.last_eval[market_id] = {"time": now - 2, "bid": 0.52, "ask": 0.55}
    state_static = LiveMarketState(bid=0.52, ask=0.55, mid=0.535, spread=0.03, depth=10, last_update=now)
    assert bot._should_evaluate(market_id, state_static)


def test_token_bucket_limiter_basic():
    limiter = TokenBucketLimiter(rate_per_sec=2, capacity=2)
    assert limiter.consume()
    assert limiter.consume()
    assert not limiter.consume()
    time.sleep(0.6)
    assert limiter.consume()


def test_backoff_progression_caps():
    random.seed(0)
    base = 100
    max_ms = 500
    d1 = compute_backoff_ms(0, base, max_ms)
    d2 = compute_backoff_ms(d1, base, max_ms)
    d3 = compute_backoff_ms(max_ms, base, max_ms)

    assert base <= d1 <= max_ms
    assert d2 >= d1
    assert d3 <= max_ms
