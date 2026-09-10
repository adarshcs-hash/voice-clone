"""Application factory.

Everything expensive is built once in the lifespan handler and attached to
``app.state``: the text frontend, the model, the database, the enrolment
service, the watermarker, the moderator and the rate limiter. Loading the model
during startup rather than on first request is what makes ``/readyz`` meaningful
-- a replica reports itself unready until it can actually serve, so a rolling
deploy does not send traffic into a cold process.

The factory takes optional overrides so tests can substitute a component
without monkey-patching module state.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Final

from fastapi import FastAPI

from mlvoice.__version__ import __version__
from mlvoice.api.middleware import RequestContextMiddleware, install_exception_handlers
from mlvoice.api.ratelimit import InMemoryRateLimiter, RateLimiter
from mlvoice.api.routes import health, tts, ui, voices
from mlvoice.asr import build_transcriber
from mlvoice.config import Settings, get_settings
from mlvoice.errors import ConfigurationError
from mlvoice.logging import configure_logging, get_logger
from mlvoice.protocols import Transcriber
from mlvoice.safety.moderation import Moderator, NameBlocklist, PatternModerator
from mlvoice.safety.watermark import SpreadSpectrumWatermarker, Watermarker
from mlvoice.synthesis import SynthesisService
from mlvoice.text.chunker import ChunkConfig
from mlvoice.text.codemix import CodeMixConfig, ManglishDetection
from mlvoice.text.g2p import G2PConfig
from mlvoice.text.pipeline import TextPipeline, TextPipelineConfig
from mlvoice.tts.base import Synthesizer
from mlvoice.tts.registry import build_synthesizer
from mlvoice.voices.consent import ConsentVerifier
from mlvoice.voices.enrollment import EnrollmentService
from mlvoice.voices.store import VoiceStore

__all__ = ["Overrides", "create_app"]

log = get_logger(__name__)

_DESCRIPTION = """
Malayalam text-to-speech and voice cloning.

**Text frontend.** Input is Unicode-normalised (legacy chillu and `nta`
spellings repaired), numbers, dates, currency and units are expanded into
Malayalam words, romanised Malayalam is transliterated, acronyms are spelled
out, and the result is chunked at sentence and clause boundaries. `POST
/v1/text/analyze` exposes every stage.

**Voice cloning requires verified consent.** Enrolment is a two-step spoken
challenge: the subject reads a phrase containing their name, the date and a
random nonce, and the service verifies both that the phrase was spoken and that
it is the same speaker as the reference clip. Revoking consent disables every
voice that relied on it.

**Generated audio is watermarked**, so a clip can be traced back to the request
that produced it via `POST /v1/watermark/detect`.
"""


@dataclass(slots=True)
class Overrides:
    """Component substitutions, for tests and for embedding the app.

    Every field defaults to ``None``, meaning "build the real thing from
    settings".
    """

    synthesizer: Synthesizer | None = None
    transcriber: Transcriber | None = None
    store: VoiceStore | None = None
    pipeline: TextPipeline | None = None
    watermarker: Watermarker | None = None
    moderator: Moderator | None = None
    rate_limiter: RateLimiter | None = None
    consent_verifier: ConsentVerifier | None = None
    settings: Settings | None = None
    extra_state: dict[str, Any] = field(default_factory=dict)


def _build_pipeline(settings: Settings) -> TextPipeline:
    detection = (
        ManglishDetection.EVIDENCE if settings.transliterate_latin else ManglishDetection.OFF
    )
    return TextPipeline(
        TextPipelineConfig(
            max_chars=settings.max_chars_per_request,
            codemix=CodeMixConfig(manglish_detection=detection),
            g2p=G2PConfig(intervocalic_voicing=settings.apply_intervocalic_voicing),
            chunking=ChunkConfig(),
        )
    )


def _build_watermarker(settings: Settings) -> Watermarker | None:
    if not settings.watermark_enabled:
        log.warning("watermarking is disabled; generated audio has no provenance marker")
        return None
    key = settings.watermark_key.get_secret_value().encode()
    if not key:
        # Production validation rejects this combination at startup; outside
        # production a development key keeps the code path exercised.
        log.warning("no watermark key configured; using a development key")
        key = b"mlvoice-development-watermark-key"
    return SpreadSpectrumWatermarker(key)


_WARMUP_TEXT: Final = "ഒന്ന് രണ്ട് മൂന്ന്."


def _load_transcriber(transcriber: Transcriber | None, settings: Settings) -> Transcriber | None:
    """Prepare the transcriber, loading it now only when that is the right cost.

    Recognition is an optional runtime: it needs the ``models`` extra, which a
    deployment serving only the text frontend has no reason to install. Taking
    the whole process down because an optional dependency is absent would be
    wrong, so a load failure disables transcription and says so. Callers that
    need it then fail at request time with a specific error, which is where the
    caller can act on it.

    Loading it during startup is off by default, because doing it cost a real
    outage: on a cold cache the default Whisper model is a three-gigabyte
    download, and pulling it inside the lifespan handler left the process stuck
    before it bound a socket -- no API, no web client, no ``/healthz`` -- while
    an *optional* feature fetched weights behind a progress bar with no log line
    to explain it. Deferring the load moves that cost onto the first
    transcription, where the caller who asked for it is the one waiting.

    Production with consent enabled is the exception and still loads eagerly.
    It mandates consent, consent verification needs recognition to confirm the
    phrase was spoken, and a replica that accepts traffic while unable to
    verify consent is worse than one that refuses to start.
    """
    if transcriber is None:
        return None
    loader = getattr(transcriber, "load", None)
    if loader is None:
        return transcriber
    required = settings.is_production and settings.require_consent
    if not (required or settings.asr_eager_load):
        log.info(
            "transcription will load on first use",
            model_id=settings.asr_model_id,
            hint=(
                "the first request pays for the download; set "
                "MLVOICE_ASR_EAGER_LOAD=true to pay it at startup instead"
            ),
        )
        return transcriber
    try:
        loader()
    except Exception as exc:
        if required:
            raise ConfigurationError(
                "consent verification requires transcription, which failed to "
                "load; install the 'models' extra or set a reachable "
                "MLVOICE_ASR_MODEL_ID",
                reason=str(exc),
            ) from exc
        log.warning(
            "transcription unavailable; reference transcripts must be supplied "
            "and consent cannot be verified",
            reason=str(exc),
        )
        return None
    return transcriber


def _warm_up(service: SynthesisService) -> None:
    """Run one synthesis through the real path during startup.

    Several dependencies import lazily on first use -- ``pyloudnorm`` pulls in
    SciPy, and NumPy plans its first FFT -- so without this the *user's* first
    request pays for it. Measured at 1.0 s on Linux and appreciably worse on
    macOS, where first load of a signed dylib also goes through Gatekeeper.
    Paying it at startup is the same principle as loading model weights before
    accepting traffic.

    It also exercises the whole chain (text frontend, synthesis, loudness,
    watermark) at boot, so a broken configuration shows up in the startup log
    rather than in the first user response.

    Non-fatal: readiness is gated on the backend having loaded, and a warm-up
    failure must not block a deploy on its own.
    """
    started = time.perf_counter()
    try:
        outcome = service.synthesize(
            _WARMUP_TEXT, owner_id="startup-warmup", request_id="startup-warmup"
        )
    except Exception as exc:
        log.warning("warm-up synthesis failed", reason=str(exc))
        return
    log.info(
        "warm-up complete",
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        audio_seconds=round(outcome.audio.duration_seconds, 3),
        watermarked=outcome.watermark_payload is not None,
    )


def create_app(overrides: Overrides | None = None) -> FastAPI:
    """Build the ASGI application.

    Args:
        overrides: Components to substitute instead of building from settings.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    overrides = overrides or Overrides()
    settings = overrides.settings or get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)

    @contextlib.asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.settings = settings
        application.state.pipeline = overrides.pipeline or _build_pipeline(settings)
        application.state.store = overrides.store or VoiceStore(settings.database_url)
        application.state.watermarker = (
            overrides.watermarker
            if overrides.watermarker is not None
            else _build_watermarker(settings)
        )
        application.state.moderator = overrides.moderator or PatternModerator()
        application.state.rate_limiter = overrides.rate_limiter or InMemoryRateLimiter(
            requests_per_minute=settings.rate_limit_per_minute
        )

        blocklist = (
            NameBlocklist.from_file(settings.blocked_voice_names_file)
            if settings.blocked_voice_names_file
            else NameBlocklist()
        )

        # Recognition serves two jobs: transcribing a reference clip so callers
        # need not type it, and verifying that a consent phrase was actually
        # spoken. Consent verification has been unusable without it.
        transcriber = (
            overrides.transcriber
            if overrides.transcriber is not None
            else build_transcriber(settings)
        )
        transcriber = _load_transcriber(transcriber, settings)
        application.state.transcriber = transcriber

        verifier = overrides.consent_verifier
        if verifier is None and transcriber is not None:
            verifier = ConsentVerifier(transcriber=transcriber)
        application.state.enrollment = EnrollmentService(
            settings,
            application.state.store,
            verifier=verifier,
            blocklist=blocklist,
            transcriber=transcriber,
        )

        synthesizer = overrides.synthesizer or build_synthesizer(settings)
        # Load before serving: a weight-loading failure must stop the rollout,
        # not surface as a 500 on the first real request.
        synthesizer.load()
        application.state.synthesizer = synthesizer

        application.state.synthesis = SynthesisService(
            settings,
            application.state.pipeline,
            synthesizer,
            application.state.store,
            moderator=application.state.moderator,
            watermarker=application.state.watermarker,
        )
        for key, value in overrides.extra_state.items():
            setattr(application.state, key, value)

        _warm_up(application.state.synthesis)

        log.info(
            "service ready",
            version=__version__,
            environment=settings.env.value,
            backend=synthesizer.info.backend,
            model_id=synthesizer.info.model_id,
            watermarking=application.state.watermarker is not None,
            auth_enabled=bool(settings.parsed_api_keys),
            transcription=transcriber is not None,
        )
        yield
        log.info("service shutting down")

    app = FastAPI(
        title="mlvoice",
        version=__version__,
        description=_DESCRIPTION,
        lifespan=lifespan,
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url=None,
    )
    app.add_middleware(RequestContextMiddleware)
    install_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(tts.router)
    app.include_router(voices.router)
    app.include_router(ui.router)
    return app
