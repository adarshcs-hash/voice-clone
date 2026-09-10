"""Dependency wiring.

Expensive, shared objects -- the model, the database engine, the text frontend --
are built once during application startup and stored on ``app.state``; the
providers here hand them to route functions. That keeps routes free of
construction logic and lets tests override any single dependency without
rebuilding the app.

Authentication is a static API key list. It is deliberately the simplest thing
that is not wrong: keys are compared in constant time, the owner id is derived
from the key by hash so one tenant's voices are invisible to another, and the
key itself is never logged. A real deployment fronts this with its identity
provider; the seam is :func:`require_api_key`.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, Request

from mlvoice.config import Settings
from mlvoice.errors import AuthenticationError, RateLimitedError
from mlvoice.protocols import Transcriber
from mlvoice.safety.watermark import Watermarker
from mlvoice.text.pipeline import TextPipeline
from mlvoice.tts.base import Synthesizer
from mlvoice.voices.enrollment import EnrollmentService
from mlvoice.voices.store import VoiceStore

__all__ = [
    "Caller",
    "get_enrollment_service",
    "get_pipeline",
    "get_settings_dep",
    "get_store",
    "get_synthesizer",
    "get_transcriber",
    "get_watermarker",
    "rate_limit",
    "require_api_key",
]


@dataclass(frozen=True, slots=True)
class Caller:
    """The authenticated caller.

    Attributes:
        owner_id: Derived from the API key. Scopes every voice operation.
        anonymous: True when authentication is disabled (development only).
    """

    owner_id: str
    anonymous: bool = False


def get_settings_dep(request: Request) -> Settings:
    """Return the process settings."""
    settings: Settings = request.app.state.settings
    return settings


def get_pipeline(request: Request) -> TextPipeline:
    """Return the shared text frontend."""
    pipeline: TextPipeline = request.app.state.pipeline
    return pipeline


def get_synthesizer(request: Request) -> Synthesizer:
    """Return the loaded synthesis backend."""
    synthesizer: Synthesizer = request.app.state.synthesizer
    return synthesizer


def get_store(request: Request) -> VoiceStore:
    """Return the voice store."""
    store: VoiceStore = request.app.state.store
    return store


def get_enrollment_service(request: Request) -> EnrollmentService:
    """Return the enrolment service."""
    service: EnrollmentService = request.app.state.enrollment
    return service


def get_transcriber(request: Request) -> Transcriber | None:
    """Return the transcriber, or ``None`` when recognition is disabled."""
    transcriber: Transcriber | None = request.app.state.transcriber
    return transcriber


def get_watermarker(request: Request) -> Watermarker | None:
    """Return the watermarker, or ``None`` when watermarking is disabled."""
    watermarker: Watermarker | None = request.app.state.watermarker
    return watermarker


def _owner_id_for(api_key: str) -> str:
    """Derive a stable, non-reversible owner id from an API key."""
    return hashlib.sha256(f"owner:{api_key}".encode()).hexdigest()[:24]


def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> Caller:
    """Authenticate the caller.

    Raises:
        AuthenticationError: The key is missing or unrecognised. The same error
            is returned for both, so the endpoint does not confirm which keys
            exist.
    """
    settings: Settings = request.app.state.settings
    accepted = settings.parsed_api_keys
    if not accepted:
        # Startup refuses this combination in production; here it means a
        # developer is running without auth on purpose.
        return Caller(owner_id="local-development", anonymous=True)
    if x_api_key is None:
        raise AuthenticationError("X-API-Key header is required")
    for candidate in accepted:
        if hmac.compare_digest(x_api_key, candidate):
            return Caller(owner_id=_owner_id_for(candidate))
    raise AuthenticationError("invalid API key")


def _request_cost(request: Request, settings: Settings) -> int:
    """Cost this request should draw from the caller's character budget.

    Only JSON bodies are charged by size: for those, content length is a good
    proxy for the text about to be synthesised, so a caller sending novels
    consumes proportionally more budget. Multipart uploads are charged as a
    single unit -- their size is audio bytes, not characters, and charging a
    700 kB reference clip against a character budget would make enrolment
    impossible.

    The cost is capped at the per-request character limit so that a bogus
    ``Content-Length`` cannot lock a caller out.
    """
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        return 1
    try:
        length = int(request.headers.get("content-length") or 1)
    except ValueError:
        return 1
    return max(1, min(length, settings.max_chars_per_request))


def rate_limit(
    request: Request,
    caller: Annotated[Caller, Depends(require_api_key)],
) -> Caller:
    """Apply the rate limit for this caller.

    Raises:
        RateLimitedError: The caller is over budget.
    """
    limiter = request.app.state.rate_limiter
    settings: Settings = request.app.state.settings
    decision = limiter.check(caller.owner_id, cost=_request_cost(request, settings))
    if not decision.allowed:
        raise RateLimitedError(
            decision.reason or "rate limit exceeded",
            retry_after_seconds=decision.retry_after_seconds,
        )
    return caller
