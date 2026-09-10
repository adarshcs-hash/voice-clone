"""Typed error hierarchy.

Every error raised deliberately by this package inherits from :class:`MlvoiceError`
and carries a stable machine-readable ``code``. The API layer maps ``code`` to an
HTTP status once, in one place, so that error contracts cannot drift between
endpoints.
"""

from __future__ import annotations

from typing import Any


class MlvoiceError(Exception):
    """Base class for all deliberate failures in this package."""

    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, message: str, /, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context

    def to_dict(self) -> dict[str, Any]:
        """Render as the JSON body the API returns."""
        body: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.context:
            body["context"] = self.context
        return body

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


class ConfigurationError(MlvoiceError):
    """The process is misconfigured and cannot serve traffic safely."""

    code = "configuration_error"
    http_status = 500


class FeatureDisabledError(MlvoiceError):
    """An optional feature is switched off in this deployment.

    404 rather than 403: a disabled feature should be indistinguishable from
    one that was never deployed, so that turning the web client off does not
    advertise that a web client exists.
    """

    code = "feature_disabled"
    http_status = 404


class ValidationError(MlvoiceError):
    """Caller-supplied input is malformed or out of bounds."""

    code = "validation_error"
    http_status = 422


class TextTooLongError(ValidationError):
    """Input text exceeds the configured per-request budget."""

    code = "text_too_long"
    http_status = 413


class UnsupportedScriptError(ValidationError):
    """Input contains a script this pipeline will not synthesise."""

    code = "unsupported_script"


class VoiceNotFoundError(MlvoiceError):
    """No enrolled voice matches the requested identifier."""

    code = "voice_not_found"
    http_status = 404


class ConsentError(MlvoiceError):
    """Voice enrolment lacks a valid consent record."""

    code = "consent_required"
    http_status = 403


class ModerationError(MlvoiceError):
    """Request was refused by the safety layer."""

    code = "moderation_blocked"
    http_status = 403


class AudioError(MlvoiceError):
    """Audio could not be decoded, or fails an ingest quality gate."""

    code = "audio_error"
    http_status = 422


class BackendUnavailableError(MlvoiceError):
    """The synthesis backend is not loaded or not reachable."""

    code = "backend_unavailable"
    http_status = 503


class RateLimitedError(MlvoiceError):
    """Caller exceeded its request budget."""

    code = "rate_limited"
    http_status = 429


class AuthenticationError(MlvoiceError):
    """Missing or invalid credentials."""

    code = "unauthenticated"
    http_status = 401
