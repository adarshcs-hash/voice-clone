"""Rate limiting."""

from __future__ import annotations

from mlvoice.api.ratelimit import InMemoryRateLimiter


class TestRequestBudget:
    def test_burst_is_allowed_then_refused(self) -> None:
        limiter = InMemoryRateLimiter(requests_per_minute=6, characters_per_minute=100_000)
        allowed = sum(1 for _ in range(20) if limiter.check("key").allowed)
        assert allowed == 9  # 6 per minute x the 1.5 burst multiplier

    def test_rejection_reports_a_retry_delay(self) -> None:
        limiter = InMemoryRateLimiter(requests_per_minute=1, characters_per_minute=100_000)
        for _ in range(5):
            limiter.check("key")
        decision = limiter.check("key")
        assert not decision.allowed
        assert decision.retry_after_seconds > 0
        assert decision.reason == "request rate exceeded"

    def test_keys_are_independent(self) -> None:
        limiter = InMemoryRateLimiter(requests_per_minute=1, characters_per_minute=100_000)
        for _ in range(5):
            limiter.check("a")
        assert limiter.check("b").allowed


class TestCharacterBudget:
    def test_character_heavy_requests_are_limited(self) -> None:
        limiter = InMemoryRateLimiter(requests_per_minute=1_000, characters_per_minute=100)
        decision = limiter.check("key", cost=500)
        assert not decision.allowed
        assert decision.reason == "character rate exceeded"

    def test_a_character_rejection_does_not_consume_a_request_token(self) -> None:
        limiter = InMemoryRateLimiter(requests_per_minute=2, characters_per_minute=50)
        for _ in range(10):
            limiter.check("key", cost=1_000)  # all refused on characters
        # The request budget must be untouched, so a small request still passes.
        assert limiter.check("key", cost=1).allowed

    def test_default_character_budget_scales_with_requests(self) -> None:
        limiter = InMemoryRateLimiter(requests_per_minute=60)
        assert limiter.check("key", cost=2_000).allowed


class TestRefill:
    def test_tokens_refill_over_time(self, monkeypatch) -> None:
        clock = {"now": 1_000.0}
        monkeypatch.setattr("mlvoice.api.ratelimit.time.monotonic", lambda: clock["now"])
        limiter = InMemoryRateLimiter(requests_per_minute=60, characters_per_minute=100_000)
        while limiter.check("key").allowed:
            pass
        clock["now"] += 60.0
        assert limiter.check("key").allowed

    def test_remaining_is_reported(self) -> None:
        limiter = InMemoryRateLimiter(requests_per_minute=60)
        assert limiter.check("key").remaining > 0
