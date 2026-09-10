"""Process configuration.

Settings come from the environment (12-factor), are validated once at startup
and are then immutable. ``get_settings`` is cached so that every layer sees the
same object; tests override it through the FastAPI dependency, not by mutating
globals.
"""

from __future__ import annotations

import re
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mlvoice.errors import ConfigurationError


class Environment(StrEnum):
    """Deployment environment. Controls which safety defaults are mandatory."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Validated runtime configuration."""

    model_config = SettingsConfigDict(
        env_prefix="MLVOICE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # -- process ------------------------------------------------------------
    env: Environment = Environment.DEVELOPMENT
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    service_name: str = "mlvoice"

    # -- auth ---------------------------------------------------------------
    api_keys: SecretStr = SecretStr("")
    """Comma-separated API keys. Empty is only tolerated outside production."""

    # -- synthesis ----------------------------------------------------------
    tts_backend: str = "dummy"
    model_id: str = "ai4bharat/IndicF5"
    model_revision: str | None = None
    """Pin the weights revision. Unpinned weights are refused in production."""
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    sample_rate: int = Field(default=24_000, ge=8_000, le=48_000)
    max_chars_per_request: int = Field(default=5_000, ge=1, le=100_000)
    synth_timeout_seconds: float = Field(default=60.0, gt=0)

    # -- transcription ------------------------------------------------------
    asr_enabled: bool = True
    """Transcribe a reference clip when no transcript is supplied.

    Without this, a caller must type the transcript of their own recording,
    which is both poor product and the main source of bad clones: a wrong
    transcript clones the voice correctly and garbles the words."""
    asr_model_id: str = "openai/whisper-large-v3"
    """Recogniser for reference clips. An Indic-specific model is materially
    better on Malayalam; see :mod:`mlvoice.asr` for candidates."""
    asr_revision: str | None = None
    asr_language: str | None = "ml"
    """Language hint. Whisper detects Malayalam unreliably on the short clips
    references always are, so the hint matters."""
    asr_trust_remote_code: bool = False
    """Required by IndicConformer and models like it."""

    # -- text frontend ------------------------------------------------------
    apply_intervocalic_voicing: bool = False
    """Allophonic voicing of intervocalic stops. Dialect- and register-
    dependent, so off by default: a neural acoustic model trained on real speech
    learns the variation better than a rule can impose it. See
    ``docs/malayalam-linguistics.md``."""
    transliterate_latin: bool = True
    """Route Latin-script runs through the Manglish transliterator. When off,
    Latin text is passed through untouched and only the loanword lexicon and
    acronym spelling apply."""

    # -- storage ------------------------------------------------------------
    database_url: str = "sqlite:///./data/mlvoice.db"
    voice_storage_dir: Path = Path("./data/voices")

    # -- limits -------------------------------------------------------------
    rate_limit_per_minute: int = Field(default=60, ge=1)
    max_reference_audio_seconds: float = Field(default=120.0, gt=0)
    min_reference_audio_seconds: float = Field(default=3.0, gt=0)

    # -- safety -------------------------------------------------------------
    consent_signing_key: SecretStr = SecretStr("")
    """HMAC key for consent challenge tokens. Required in production."""
    watermark_key: SecretStr = SecretStr("")
    """HMAC key for the spread-spectrum watermark. Required in production, and
    must not be rotated casually: rotating it makes previously generated audio
    undetectable."""
    require_consent: bool = True
    watermark_enabled: bool = True
    blocked_voice_names_file: Path | None = None
    """Newline-delimited names refused at enrolment (public figures)."""

    # -- web ui -------------------------------------------------------------
    ui_enabled: bool = True
    """Serve the browser client at ``/ui``.

    The page itself holds no secrets and every call it makes still needs an API
    key, but a public deployment may prefer not to advertise a cloning console
    at a guessable path."""

    @field_validator("api_keys")
    @classmethod
    def _strip_keys(cls, value: SecretStr) -> SecretStr:
        return SecretStr(value.get_secret_value().strip())

    @field_validator("model_revision", "asr_revision", "blocked_voice_names_file", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """Treat a blank environment variable as unset.

        ``MLVOICE_MODEL_REVISION=`` in a ``.env`` file arrives as the empty
        string, not as absent. Without this, an optional ``Path`` field becomes
        ``Path("")`` -- which is ``Path(".")``, is truthy, and exists -- and an
        optional ``str`` field becomes ``""``, which is not ``None`` and would
        satisfy an ``is None`` check. Both were real failures: the first crashed
        startup by reading the working directory as a blocklist, the second let
        an unpinned model revision through the production gate.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _enforce_production_invariants(self) -> Settings:
        if self.env is not Environment.PRODUCTION:
            return self
        problems: list[str] = []
        if not self.parsed_api_keys:
            problems.append("MLVOICE_API_KEYS must be set in production")
        if not self.require_consent:
            problems.append("MLVOICE_REQUIRE_CONSENT cannot be disabled in production")
        if not self.watermark_enabled:
            problems.append("MLVOICE_WATERMARK_ENABLED cannot be disabled in production")
        if not self.consent_signing_key.get_secret_value():
            problems.append("MLVOICE_CONSENT_SIGNING_KEY must be set in production")
        if self.watermark_enabled and not self.watermark_key.get_secret_value():
            problems.append("MLVOICE_WATERMARK_KEY must be set when watermarking is enabled")
        if self.tts_backend == "dummy":
            problems.append("the dummy backend cannot serve production traffic")
        # Falsiness, not ``is None``: a blank value must not satisfy this gate.
        if not self.model_revision and self.tts_backend != "dummy":
            problems.append("MLVOICE_MODEL_REVISION must pin the weights in production")
        if problems:
            raise ConfigurationError("invalid production configuration", problems=problems)
        return self

    @property
    def parsed_api_keys(self) -> frozenset[str]:
        """API keys as a set. Empty means auth is disabled."""
        raw = self.api_keys.get_secret_value()
        return frozenset(k for k in re.split(r"[,\s]+", raw) if k)

    @property
    def is_production(self) -> bool:
        return self.env is Environment.PRODUCTION


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, constructed once."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache. For tests and for CLI flag overrides only."""
    get_settings.cache_clear()
