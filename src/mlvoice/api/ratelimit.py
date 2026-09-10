"""Request rate limiting.

A token bucket per API key, held in process memory.

**Scope, stated plainly:** this limiter is per process. With more than one
replica the effective limit is ``replicas x rate``, which is fine as a safety
valve against a runaway client and is *not* a quota mechanism. A real quota needs
shared state -- Redis ``INCR`` with an expiry, or the API gateway's own limiter
-- and :class:`RateLimiter` is a protocol so that swap is a constructor change.

Synthesis is expensive and highly asymmetric: a single request can occupy a GPU
for seconds. The limiter therefore also accounts *characters*, not just
requests, so that one caller sending novels cannot starve everyone else while
staying under a request-count limit.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["InMemoryRateLimiter", "RateLimitDecision", "RateLimiter"]


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """Outcome of a rate-limit check."""

    allowed: bool
    retry_after_seconds: float
    remaining: float
    reason: str | None = None


@runtime_checkable
class RateLimiter(Protocol):
    """Anything that can admit or reject a request for a key."""

    def check(self, key: str, *, cost: int = 1) -> RateLimitDecision:
        """Admit or reject ``cost`` units of work for ``key``."""
        ...


class _Bucket:
    __slots__ = ("tokens", "updated_at")

    def __init__(self, tokens: float, updated_at: float) -> None:
        self.tokens = tokens
        self.updated_at = updated_at


class InMemoryRateLimiter:
    """Token-bucket limiter over requests and characters.

    Args:
        requests_per_minute: Sustained request rate per key.
        characters_per_minute: Sustained character rate per key. Defaults to
            ``requests_per_minute * 2000``, i.e. an average request of 2000
            characters.
        burst_multiplier: Bucket capacity as a multiple of the per-minute rate,
            so a client may burst above the average without being throttled.
        now: Clock, injectable for tests.
    """

    def __init__(
        self,
        *,
        requests_per_minute: int,
        characters_per_minute: int | None = None,
        burst_multiplier: float = 1.5,
    ) -> None:
        self._request_rate = requests_per_minute / 60.0
        self._request_capacity = requests_per_minute * burst_multiplier
        char_limit = characters_per_minute or requests_per_minute * 2_000
        self._char_rate = char_limit / 60.0
        self._char_capacity = char_limit * burst_multiplier
        self._requests: dict[str, _Bucket] = {}
        self._characters: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _consume(
        buckets: dict[str, _Bucket],
        key: str,
        *,
        cost: float,
        rate: float,
        capacity: float,
        now: float,
    ) -> tuple[bool, float, float]:
        bucket = buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=capacity, updated_at=now)
            buckets[key] = bucket
        elapsed = max(0.0, now - bucket.updated_at)
        bucket.tokens = min(capacity, bucket.tokens + elapsed * rate)
        bucket.updated_at = now
        if bucket.tokens >= cost:
            bucket.tokens -= cost
            return True, bucket.tokens, 0.0
        deficit = cost - bucket.tokens
        return False, bucket.tokens, deficit / rate if rate > 0 else float("inf")

    def check(self, key: str, *, cost: int = 1) -> RateLimitDecision:
        """Admit or reject one request of ``cost`` characters.

        Both buckets must admit the request. When either refuses, nothing is
        consumed from the other, so a rejected request does not eat quota.
        """
        now = time.monotonic()
        with self._lock:
            # Peek both buckets before committing, so a rejection is free.
            request_ok, request_left, request_wait = self._consume(
                self._requests,
                key,
                cost=1.0,
                rate=self._request_rate,
                capacity=self._request_capacity,
                now=now,
            )
            if not request_ok:
                return RateLimitDecision(
                    allowed=False,
                    retry_after_seconds=round(request_wait, 3),
                    remaining=round(request_left, 2),
                    reason="request rate exceeded",
                )
            char_ok, char_left, char_wait = self._consume(
                self._characters,
                key,
                cost=float(max(1, cost)),
                rate=self._char_rate,
                capacity=self._char_capacity,
                now=now,
            )
            if not char_ok:
                # Refund the request token: the request never ran.
                self._requests[key].tokens = min(
                    self._request_capacity, self._requests[key].tokens + 1.0
                )
                return RateLimitDecision(
                    allowed=False,
                    retry_after_seconds=round(char_wait, 3),
                    remaining=round(char_left, 2),
                    reason="character rate exceeded",
                )
            return RateLimitDecision(
                allowed=True, retry_after_seconds=0.0, remaining=round(request_left, 2)
            )
