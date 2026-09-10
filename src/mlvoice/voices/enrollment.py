"""Voice enrolment.

Enrolment is the only place a new voice enters the system, which makes it the
only place the safety invariants can be enforced. The order matters: cheap
refusals first, then audio validation, then consent, and persistence last, so
that a rejected enrolment leaves nothing behind.

::

    name blocklist -> reference audio validation -> consent verification
                   -> persist audio -> persist consent -> persist voice

The consent challenge is a **signed token**, not server-side state. The service
issues ``(phrase, token)``; the client returns the token with the recording; the
service re-derives the phrase from the token and checks the signature and age.
That keeps enrolment stateless across replicas without letting a client forge or
replay a challenge.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from mlvoice.audio.io import Audio, load_audio, save_audio
from mlvoice.audio.loudness import BROADCAST_TARGET_LUFS, normalize_loudness
from mlvoice.audio.quality import QualityGate, measure_quality
from mlvoice.audio.vad import trim_silence
from mlvoice.config import Settings
from mlvoice.errors import AudioError, ConsentError, ModerationError, ValidationError
from mlvoice.logging import get_logger
from mlvoice.protocols import Transcriber
from mlvoice.safety.moderation import NameBlocklist
from mlvoice.voices.consent import (
    DEFAULT_VALIDITY,
    ConsentPhrase,
    ConsentRecord,
    ConsentStatus,
    ConsentVerifier,
    issue_phrase,
)
from mlvoice.voices.store import Voice, VoiceStatus, VoiceStore

__all__ = ["EnrollmentRequest", "EnrollmentService", "IssuedChallenge"]

log = get_logger(__name__)

_TOKEN_VERSION: Final = "v1"  # noqa: S105 - a schema version, not a secret
_CHALLENGE_TTL: Final = timedelta(hours=1)


@dataclass(frozen=True, slots=True)
class IssuedChallenge:
    """A consent challenge and the signed token that authenticates it."""

    phrase: ConsentPhrase
    token: str


@dataclass(frozen=True, slots=True)
class EnrollmentRequest:
    """Everything needed to enrol one voice.

    Attributes:
        name: Display name for the voice.
        owner_id: Tenant or account that will own it.
        reference_audio: Encoded audio of the voice to clone.
        reference_text: Verbatim Malayalam transcript of ``reference_audio``,
            or ``None`` to have it transcribed. The cloning backend conditions
            on this, and a wrong transcript is the most common cause of a poor
            clone -- which is why asking a caller to type it is worse than
            recognising it and letting them correct the result.
        consent_token: Token from :meth:`EnrollmentService.issue_challenge`.
        consent_audio: Encoded recording of the subject reading the challenge.
        dialect: Optional regional label, for corpus and catalogue reporting.
        gender: Optional label, for catalogue reporting.
    """

    name: str
    owner_id: str
    reference_audio: bytes
    reference_text: str | None
    consent_token: str
    consent_audio: bytes
    dialect: str = "unknown"
    gender: str = "unknown"


class EnrollmentService:
    """Enrols voices, enforcing the safety invariants.

    Args:
        settings: Process configuration; supplies the signing key, the storage
            directory and the reference-duration bounds.
        store: Persistence.
        verifier: Consent verifier. Required whenever
            ``settings.require_consent`` is set, which production enforces.
        blocklist: Public-figure name blocklist.
        quality_gate: Objective thresholds the reference clip must meet.
        transcriber: Used when a request supplies no reference transcript.
            Without one, a transcript becomes mandatory.
    """

    def __init__(
        self,
        settings: Settings,
        store: VoiceStore,
        *,
        verifier: ConsentVerifier | None = None,
        blocklist: NameBlocklist | None = None,
        quality_gate: QualityGate | None = None,
        transcriber: Transcriber | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._verifier = verifier
        self._transcriber = transcriber
        self._blocklist = blocklist or NameBlocklist()
        # Reference clips are short, so the corpus silence gate does not apply.
        self._gate = quality_gate or QualityGate(
            min_duration_seconds=settings.min_reference_audio_seconds,
            max_duration_seconds=settings.max_reference_audio_seconds,
            min_snr_db=15.0,
            max_silence_ratio=0.8,
        )
        self._key = settings.consent_signing_key.get_secret_value().encode()

    # -- challenge ----------------------------------------------------------

    def issue_challenge(self, subject_name: str) -> IssuedChallenge:
        """Issue a consent challenge and its signed token."""
        phrase = issue_phrase(subject_name)
        payload = {
            "v": _TOKEN_VERSION,
            "subject": phrase.subject_name,
            "text": phrase.text,
            "nonce": list(phrase.nonce),
            "issued_at": phrase.issued_at.isoformat(),
        }
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        signature = hmac.new(self._key, body, hashlib.sha256).digest()
        token = (
            base64.urlsafe_b64encode(body).decode().rstrip("=")
            + "."
            + base64.urlsafe_b64encode(signature).decode().rstrip("=")
        )
        return IssuedChallenge(phrase=phrase, token=token)

    def _decode_challenge(self, token: str) -> ConsentPhrase:
        """Recover the challenge from a token, verifying signature and age.

        Raises:
            ConsentError: The token is malformed, unsigned, forged or expired.
        """
        if not self._key:
            raise ConsentError("consent signing key is not configured")
        try:
            body_b64, signature_b64 = token.split(".", 1)
            body = base64.urlsafe_b64decode(_pad(body_b64))
            signature = base64.urlsafe_b64decode(_pad(signature_b64))
        except (ValueError, TypeError) as exc:
            raise ConsentError("malformed consent token") from exc

        expected = hmac.new(self._key, body, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise ConsentError("consent token signature does not verify")

        try:
            payload = json.loads(body)
            phrase = ConsentPhrase(
                text=payload["text"],
                nonce=tuple(payload["nonce"]),
                issued_at=datetime.fromisoformat(payload["issued_at"]),
                subject_name=payload["subject"],
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ConsentError("consent token payload is invalid") from exc

        if not phrase.is_fresh(max_age=_CHALLENGE_TTL):
            raise ConsentError("consent challenge has expired; request a new one")
        return phrase

    # -- enrolment ----------------------------------------------------------

    def enroll(self, request: EnrollmentRequest) -> Voice:
        """Enrol a voice.

        Raises:
            ValidationError: The reference transcript is missing.
            ModerationError: The voice name is on the blocklist.
            AudioError: The reference clip fails the quality gate.
            ConsentError: Consent is required and could not be verified.
        """
        phrase = self._decode_challenge(request.consent_token)
        for candidate in (request.name, phrase.subject_name):
            if self._blocklist.is_blocked(candidate):
                log.warning("blocked enrolment name", name=candidate, owner_id=request.owner_id)
                raise ModerationError("this voice name cannot be enrolled", name=candidate)

        reference = self._prepare_reference(request.reference_audio)
        reference_text = self._resolve_transcript(request.reference_text, reference)
        consent_audio = load_audio(
            request.consent_audio, target_sample_rate=self._settings.sample_rate
        )

        if self._settings.require_consent:
            if self._verifier is None:
                raise ConsentError(
                    "consent is required but no verifier is configured; refusing "
                    "to enrol a voice without verified consent"
                )
            verification = self._verifier.verify(
                phrase, consent_audio, reference, max_challenge_age=_CHALLENGE_TTL
            )
            if not verification.passed:
                raise ConsentError(
                    "consent verification failed", reasons=list(verification.reasons)
                )
        else:
            log.warning(
                "enrolling without consent verification; not permitted in production",
                owner_id=request.owner_id,
            )
            verification = None

        voice_id = f"voice_{uuid.uuid4().hex[:16]}"
        base = Path(self._settings.voice_storage_dir) / request.owner_id / voice_id
        reference_path = save_audio(reference, base / "reference.wav")
        consent_path = save_audio(consent_audio, base / "consent.wav")

        consent_id: str | None = None
        if verification is not None:
            now = datetime.now(UTC)
            record = ConsentRecord(
                id=f"consent_{uuid.uuid4().hex[:16]}",
                voice_id=voice_id,
                subject_name=phrase.subject_name,
                phrase_text=phrase.text,
                nonce=phrase.nonce,
                status=ConsentStatus.VERIFIED,
                verification=verification,
                consent_audio_path=str(consent_path),
                created_at=now,
                expires_at=now + DEFAULT_VALIDITY,
            )
            self._store.save_consent(record)
            consent_id = record.id

        now = datetime.now(UTC)
        voice = Voice(
            id=voice_id,
            name=request.name,
            owner_id=request.owner_id,
            reference_audio_path=str(reference_path),
            reference_text=reference_text,
            reference_duration_seconds=reference.duration_seconds,
            sample_rate=reference.sample_rate,
            status=VoiceStatus.ACTIVE,
            consent_id=consent_id,
            dialect=request.dialect,
            gender=request.gender,
            quality=measure_quality(reference).as_dict(),
            created_at=now,
            updated_at=now,
        )
        return self._store.create_voice(voice)

    def _resolve_transcript(self, supplied: str | None, reference: Audio) -> str:
        """Return the reference transcript, recognising it when not supplied.

        Raises:
            ValidationError: No transcript was given and no transcriber is
                configured, so there is nothing to condition the clone on.
        """
        if supplied is not None and supplied.strip():
            return supplied.strip()
        if self._transcriber is None:
            raise ValidationError(
                "a reference transcript is required when transcription is "
                "unavailable: the cloning backend conditions on it, and a "
                "missing or wrong transcript degrades the clone"
            )
        recognised = self._transcriber.transcribe(reference).strip()
        if not recognised:
            raise ValidationError(
                "the reference recording could not be transcribed; it may be "
                "silent, too noisy, or not speech"
            )
        log.info(
            "reference transcript recognised",
            transcriber=getattr(self._transcriber, "name", "unknown"),
            characters=len(recognised),
        )
        return recognised

    def _prepare_reference(self, encoded: bytes) -> Audio:
        """Decode, trim, level and gate the reference clip.

        Raises:
            AudioError: The clip fails the quality gate.
        """
        raw = load_audio(encoded, target_sample_rate=self._settings.sample_rate)
        report = measure_quality(raw)
        trimmed = trim_silence(raw)
        levelled, _ = normalize_loudness(trimmed, target_lufs=BROADCAST_TARGET_LUFS)

        from dataclasses import replace

        final = measure_quality(levelled)
        combined = replace(
            report,
            duration_seconds=levelled.duration_seconds,
            silence_ratio=final.silence_ratio,
            peak_dbfs=final.peak_dbfs,
            clipping_ratio=final.clipping_ratio,
        )
        passed, reasons = self._gate.evaluate(combined)
        if not passed:
            raise AudioError("reference audio failed quality checks", reasons=reasons)
        return levelled


def _pad(value: str) -> str:
    """Restore base64url padding stripped for compactness."""
    return value + "=" * (-len(value) % 4)
