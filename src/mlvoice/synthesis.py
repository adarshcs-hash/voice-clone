"""The synthesis application service.

Composes the pieces in the order the product requires, so that no route -- and
no future caller -- can skip a step:

::

    moderate text -> run the text frontend -> resolve and authorise the voice
                  -> synthesise -> normalise loudness -> watermark

Two checks in here are easy to get wrong by omission.

**Consent is re-checked at synthesis time, not only at enrolment.** A consent
record can be revoked after a voice is created, and a revoked consent that still
produces audio is not a consent mechanism. Every request re-reads the voice's
status and its consent record.

**Watermarking happens after loudness normalisation.** Normalising afterwards
would apply a gain to the watermark too; the detector tolerates gain, but doing
it in this order keeps the embedded amplitude exactly what was chosen.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from mlvoice.audio.io import Audio, load_audio
from mlvoice.audio.loudness import STREAMING_TARGET_LUFS, normalize_loudness
from mlvoice.config import Settings
from mlvoice.errors import ConsentError, ModerationError, VoiceNotFoundError
from mlvoice.logging import get_logger
from mlvoice.safety.moderation import ModerationDecision, Moderator, Severity
from mlvoice.safety.watermark import Watermarker, payload_for
from mlvoice.text.pipeline import ProcessedText, TextPipeline
from mlvoice.tts.base import ReferencePrompt, SynthesisRequest, SynthesisResult, Synthesizer
from mlvoice.voices.store import VoiceStore

__all__ = ["SynthesisOutcome", "SynthesisService"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SynthesisOutcome:
    """Everything a caller needs about one completed synthesis."""

    audio: Audio
    result: SynthesisResult
    processed: ProcessedText
    moderation: ModerationDecision
    watermark_payload: int | None
    watermarker: str | None

    def headers(self) -> dict[str, str]:
        """Response headers describing the generation.

        Exposing duration and real-time factor per request is what lets a client
        detect a slow backend without access to server metrics.
        """
        headers = {
            "X-Audio-Duration": f"{self.audio.duration_seconds:.3f}",
            "X-Real-Time-Factor": f"{self.result.real_time_factor:.4f}",
            "X-Chunks": str(self.result.chunk_count),
            "X-Moderation": self.moderation.severity.value,
        }
        if self.watermark_payload is not None:
            headers["X-Watermark-Payload"] = str(self.watermark_payload)
        return headers


class SynthesisService:
    """Orchestrates text processing, voice authorisation, synthesis and safety.

    Args:
        settings: Process configuration.
        pipeline: Malayalam text frontend.
        synthesizer: Loaded synthesis backend.
        store: Voice store, for resolving and authorising cloned voices.
        moderator: Text moderation. ``None`` disables it, which production
            configuration should not do.
        watermarker: Watermarker. ``None`` disables watermarking, which
            production configuration refuses.
        target_lufs: Output loudness target.
    """

    def __init__(
        self,
        settings: Settings,
        pipeline: TextPipeline,
        synthesizer: Synthesizer,
        store: VoiceStore,
        *,
        moderator: Moderator | None = None,
        watermarker: Watermarker | None = None,
        target_lufs: float = STREAMING_TARGET_LUFS,
    ) -> None:
        self._settings = settings
        self._pipeline = pipeline
        self._synthesizer = synthesizer
        self._store = store
        self._moderator = moderator
        self._watermarker = watermarker
        self._target_lufs = target_lufs

    def synthesize(
        self,
        text: str,
        *,
        owner_id: str,
        voice_id: str | None = None,
        speed: float = 1.0,
        seed: int | None = None,
        request_id: str = "unknown",
    ) -> SynthesisOutcome:
        """Synthesise ``text``, applying every safety and quality step.

        Args:
            text: Raw caller text.
            owner_id: Authenticated caller, used to scope ``voice_id``.
            voice_id: Enrolled voice to clone, or ``None`` for the default voice.
            speed: Playback rate multiplier.
            seed: Sampling seed.
            request_id: Correlation id; becomes part of the watermark payload.

        Returns:
            The generated audio and its metadata.

        Raises:
            ModerationError: The text was refused by the safety layer.
            VoiceNotFoundError: ``voice_id`` is unknown or belongs elsewhere.
            ConsentError: The voice's consent has been revoked or has expired.
        """
        moderation = self._moderate(text)
        processed = self._pipeline.process(text)
        prompt = self._resolve_prompt(voice_id, owner_id) if voice_id else None

        result = self._synthesizer.synthesize(
            SynthesisRequest(text=processed, prompt=prompt, speed=speed, seed=seed)
        )
        audio = self._finalize(result.audio)
        audio, payload, watermarker_name = self._watermark(audio, request_id, voice_id)
        log.info(
            "synthesis served",
            voice_id=voice_id,
            moderation=moderation.severity.value,
            watermarked=payload is not None,
            generation=result.metrics(),
            text=processed.summary(),
        )
        return SynthesisOutcome(
            audio=audio,
            result=result,
            processed=processed,
            moderation=moderation,
            watermark_payload=payload,
            watermarker=watermarker_name,
        )

    def stream(
        self,
        text: str,
        *,
        owner_id: str,
        voice_id: str | None = None,
        speed: float = 1.0,
        seed: int | None = None,
    ) -> Iterator[Audio]:
        """Yield audio chunk by chunk, for a low time-to-first-byte response.

        Streamed chunks are level-normalised but **not** watermarked: the
        spread-spectrum payload needs the whole utterance, and watermarking each
        chunk separately would embed a detectable pattern per chunk instead of
        one per utterance. Callers that need a watermarked artefact should use
        :meth:`synthesize`; the streaming path is for interactive playback, and
        the service records that distinction in its logs.

        Raises:
            ModerationError: The text was refused by the safety layer.
        """
        self._moderate(text)
        processed = self._pipeline.process(text)
        prompt = self._resolve_prompt(voice_id, owner_id) if voice_id else None
        request = SynthesisRequest(text=processed, prompt=prompt, speed=speed, seed=seed)
        log.info(
            "synthesis streaming",
            voice_id=voice_id,
            chunks=len(processed.chunks),
            watermarked=False,
        )
        for chunk in self._synthesizer.stream(request):
            yield self._finalize(chunk)

    # -- internals ----------------------------------------------------------

    def _moderate(self, text: str) -> ModerationDecision:
        if self._moderator is None:
            return ModerationDecision(Severity.ALLOW)
        decision = self._moderator.review(text)
        if decision.blocked:
            log.warning("synthesis refused by moderation", rules=list(decision.rules))
            raise ModerationError(
                "this request was refused by the safety policy",
                rules=list(decision.rules),
            )
        if decision.severity is Severity.FLAG:
            log.warning("synthesis flagged for review", rules=list(decision.rules))
        return decision

    def _resolve_prompt(self, voice_id: str, owner_id: str) -> ReferencePrompt:
        voice = self._store.get_voice(voice_id, owner_id=owner_id)
        if not voice.is_usable:
            raise VoiceNotFoundError(
                "voice is disabled", voice_id=voice_id, status=voice.status.value
            )
        if self._settings.require_consent:
            if voice.consent_id is None:
                raise ConsentError("voice has no consent record", voice_id=voice_id)
            consent = self._store.get_consent(voice.consent_id)
            if not consent.is_active:
                raise ConsentError(
                    "consent for this voice is no longer valid",
                    voice_id=voice_id,
                    consent_status=consent.effective_status().value,
                )
        reference = load_audio(
            voice.reference_audio_path, target_sample_rate=self._settings.sample_rate
        )
        return ReferencePrompt(audio=reference, text=voice.reference_text, voice_id=voice.id)

    def _finalize(self, audio: Audio) -> Audio:
        """Level the audio, tolerating a chunk that is entirely silence."""
        try:
            levelled, _ = normalize_loudness(audio, target_lufs=self._target_lufs)
        except Exception:
            # A pause-only chunk has no loudness to normalise; pass it through.
            return audio
        return levelled

    def _watermark(
        self, audio: Audio, request_id: str, voice_id: str | None
    ) -> tuple[Audio, int | None, str | None]:
        """Embed the provenance payload, returning the audio unchanged on failure.

        A clip too short to carry the payload is not a request failure -- the
        caller still gets their audio -- but it is logged, because a deployment
        whose clips are routinely unmarkable has lost its provenance trail.
        """
        if self._watermarker is None:
            return audio, None, None
        payload = payload_for(request_id, voice_id or "default")
        try:
            marked = self._watermarker.embed(audio, payload)
        except Exception as exc:
            log.warning("watermarking skipped", reason=str(exc))
            return audio, None, None
        return marked, payload, self._watermarker.name
