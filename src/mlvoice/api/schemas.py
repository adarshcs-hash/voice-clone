"""HTTP request and response models.

Validation lives here rather than in the handlers, so that a malformed request
is rejected with a precise message before touching a model, and so the OpenAPI
document is generated from the same declarations the server enforces.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AnalyzeRequest",
    "AnalyzeResponse",
    "ChallengeRequest",
    "ChallengeResponse",
    "ChunkInfo",
    "ErrorResponse",
    "HealthResponse",
    "InfoResponse",
    "SynthesizeRequest",
    "TranscribeResponse",
    "VoiceListResponse",
    "VoiceResponse",
    "WatermarkResponse",
]

MalayalamText = Annotated[str, Field(min_length=1, max_length=100_000)]


class ErrorResponse(BaseModel):
    """Error body. Every non-2xx response has this shape."""

    code: str = Field(description="Stable machine-readable error code.")
    message: str
    context: dict[str, object] | None = None
    request_id: str | None = None


class SynthesizeRequest(BaseModel):
    """Body of ``POST /v1/tts``."""

    model_config = ConfigDict(extra="forbid")

    text: MalayalamText = Field(description="Malayalam text to speak. Manglish is accepted.")
    voice_id: str | None = Field(
        default=None,
        description="Enrolled voice to clone. Omit to use the backend's default voice.",
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    seed: int | None = Field(
        default=None,
        description="Sampling seed. Fix it to make a generation reproducible.",
    )
    audio_format: Literal["wav"] = Field(
        default="wav", description="Response encoding. Only 16-bit PCM WAV today."
    )


class ChunkInfo(BaseModel):
    """One synthesis chunk, as reported by the analyze endpoint."""

    index: int
    text: str
    break_after: str
    pause_ms: int
    phonemes: str


class AnalyzeRequest(BaseModel):
    """Body of ``POST /v1/text/analyze``."""

    model_config = ConfigDict(extra="forbid")

    text: MalayalamText


class AnalyzeResponse(BaseModel):
    """Every stage of the text frontend, for debugging a pronunciation report.

    This endpoint is why "why did the voice say that?" is answerable without
    reproducing the request locally.
    """

    original: str
    normalized: str
    expanded: str
    routed: str
    contains_malayalam: bool
    normalization_changes: int
    legacy_nta_fixed: int
    chunks: list[ChunkInfo]


class ChallengeRequest(BaseModel):
    """Body of ``POST /v1/voices/challenge``."""

    model_config = ConfigDict(extra="forbid")

    subject_name: str = Field(
        min_length=1,
        max_length=200,
        description="Name of the person whose voice will be cloned.",
    )


class ChallengeResponse(BaseModel):
    """A consent challenge for the subject to read aloud."""

    phrase: str = Field(description="Read this aloud and submit the recording.")
    nonce: list[str]
    token: str = Field(description="Return this with the enrolment request.")
    expires_in_seconds: int


class VoiceResponse(BaseModel):
    """An enrolled voice.

    The reference audio path is deliberately not exposed: it is biometric data
    and belongs behind the API, not in a client-visible field.
    """

    id: str
    name: str
    status: str
    reference_text: str
    reference_duration_seconds: float
    sample_rate: int
    dialect: str
    gender: str
    consent_id: str | None
    consent_verified: bool
    created_at: datetime


class VoiceListResponse(BaseModel):
    """Paginated voice list."""

    voices: list[VoiceResponse]
    count: int


class TranscribeResponse(BaseModel):
    """Recognised text for an uploaded clip.

    Returned so a client can *show* the transcript and let the user correct it.
    Recognition is not perfect, but a visible approximate transcript beats an
    invisible wrong one -- and a wrong reference transcript clones the voice
    correctly while garbling the words.
    """

    text: str = Field(
        description=(
            "The recognised text, or empty when recognition produced something "
            "unusable. Never a best-effort guess: this value is fed to the "
            "synthesiser as the reference transcript, where a wrong one "
            "corrupts the clone while looking like a filled-in field."
        )
    )
    usable: bool = Field(
        default=True,
        description=(
            "False when the transcript was discarded. `advisories` says why, "
            "and the caller should ask the user to type the sentence instead."
        ),
    )
    duration_seconds: float
    transcriber: str
    quality: dict[str, float] = Field(
        default_factory=dict,
        description="Objective measurements of the clip, so a client can warn "
        "about a recording that is too noisy or too short to clone from.",
    )
    advisories: list[str] = Field(
        default_factory=list,
        description="Non-fatal concerns, such as a clip long enough to slow "
        "every subsequent generation.",
    )


class WatermarkResponse(BaseModel):
    """Watermark detection result."""

    detected: bool
    payload: int | None
    confidence: float
    watermarker: str


class HealthResponse(BaseModel):
    """Liveness and readiness payload."""

    status: Literal["ok", "degraded"]
    version: str
    backend_ready: bool


class InfoResponse(BaseModel):
    """Service capabilities, for clients and for operators."""

    version: str
    environment: str
    backend: dict[str, object]
    watermarking: bool
    consent_required: bool
    auth_required: bool = Field(
        description=(
            "Whether requests need an API key. Exposed so a client can hide "
            "the field entirely rather than presenting an empty box on a "
            "deployment that has no keys configured."
        )
    )
    transcription: bool = Field(
        description=(
            "Whether the service can transcribe a reference clip. False means "
            "the caller must supply the transcript themselves."
        )
    )
    max_chars_per_request: int
