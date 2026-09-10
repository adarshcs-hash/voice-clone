"""Consent capture and verification.

A voice-cloning service must be able to prove, per voice, that the person whose
voice it is agreed to the cloning. Nothing else in the system substitutes for
that: watermarking establishes provenance after the fact, and moderation catches
templates, but only a consent record answers "was this person's voice used with
their permission?".

The scheme here is a spoken challenge:

1.  The service issues a :class:`ConsentPhrase` containing the subject's name,
    today's date and a random nonce drawn from common Malayalam words.
2.  The subject records themselves reading it.
3.  :class:`ConsentVerifier` checks two things -- that the recording *says* the
    phrase (ASR, gated on CER) and that it is the *same speaker* as the
    enrolment sample (speaker verification, gated on cosine similarity).

The nonce is what makes this more than a checkbox. Without it, a recording of
someone saying "I consent" could be lifted from anywhere; with it, the recording
must have been made after the challenge was issued, for this enrolment.

What this is not
----------------
It is not identity verification: it proves the person who recorded the
enrolment sample also recorded the consent, not who that person is. Binding to a
legal identity requires a document or government-ID check, which belongs at a
higher layer and is where a commercial deployment in India should also record
what it needs for the DPDP Act's consent-notice requirements. See
``docs/safety-and-compliance.md``.
"""

from __future__ import annotations

import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Final

from mlvoice.audio.io import Audio
from mlvoice.errors import ConsentError
from mlvoice.eval.metrics import character_error_rate
from mlvoice.logging import get_logger
from mlvoice.protocols import SpeakerVerifier, Transcriber

__all__ = [
    "ConsentPhrase",
    "ConsentRecord",
    "ConsentStatus",
    "ConsentVerification",
    "ConsentVerifier",
    "issue_phrase",
]

log = get_logger(__name__)

NONCE_WORDS: Final[tuple[str, ...]] = (
    "മാവ്",
    "പുഴ",
    "കടൽ",
    "മഴ",
    "വെയിൽ",
    "കാറ്റ്",
    "മല",
    "വയൽ",
    "തെങ്ങ്",
    "പൂവ്",
    "കിളി",
    "മീൻ",
    "നിലാവ്",
    "ചന്ദ്രൻ",
    "നക്ഷത്രം",
    "വഴി",
    "പാലം",
    "വീട്",
    "മുറ്റം",
    "കിണർ",
)
"""Everyday Malayalam nouns, chosen to be short, unambiguous and easy to read
aloud. The nonce must be speakable by a first-time reader, so it is drawn from
words rather than from an alphanumeric code."""

NONCE_WORD_COUNT: Final = 3
DEFAULT_VALIDITY: Final = timedelta(days=365)
_MAX_PHRASE_CER: Final = 0.25
_MIN_SPEAKER_SIMILARITY: Final = 0.70


class ConsentStatus(StrEnum):
    """Lifecycle state of a consent record."""

    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    REVOKED = "revoked"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class ConsentPhrase:
    """A challenge phrase the subject must read aloud.

    Attributes:
        text: The Malayalam sentence to read.
        nonce: The random words embedded in it, retained so verification can
            confirm they were spoken.
        issued_at: When the challenge was created. A recording is only accepted
            against a recent challenge.
        subject_name: Name as given by the subject.
    """

    text: str
    nonce: tuple[str, ...]
    issued_at: datetime
    subject_name: str

    def is_fresh(self, *, max_age: timedelta = timedelta(hours=1)) -> bool:
        """True if the challenge is recent enough to accept a recording against."""
        return datetime.now(UTC) - self.issued_at <= max_age


def issue_phrase(
    subject_name: str,
    *,
    words: Sequence[str] = NONCE_WORDS,
    word_count: int = NONCE_WORD_COUNT,
    today: date | None = None,
) -> ConsentPhrase:
    """Issue a fresh consent challenge for ``subject_name``.

    Args:
        subject_name: The subject's name, read out as part of the phrase.
        words: Nonce vocabulary.
        word_count: How many nonce words to include.
        today: Date embedded in the phrase; defaults to the current UTC date.

    Returns:
        The challenge to present to the subject.

    Raises:
        ConsentError: ``subject_name`` is empty.
    """
    if not subject_name.strip():
        raise ConsentError("a consent phrase requires the subject's name")
    chosen = tuple(secrets.choice(words) for _ in range(word_count))
    when = today or datetime.now(UTC).date()
    text = (
        f"ഞാൻ {subject_name.strip()}. "
        f"{when.day}-{when.month}-{when.year} തീയതി, "
        "എന്റെ ശബ്ദം ഉപയോഗിച്ച് കൃത്രിമ ശബ്ദം നിർമ്മിക്കാൻ ഞാൻ അനുമതി നൽകുന്നു. "
        f"സ്ഥിരീകരണ വാക്കുകൾ: {', '.join(chosen)}."
    )
    return ConsentPhrase(
        text=text, nonce=chosen, issued_at=datetime.now(UTC), subject_name=subject_name.strip()
    )


@dataclass(frozen=True, slots=True)
class ConsentVerification:
    """Result of checking a consent recording."""

    passed: bool
    phrase_cer: float
    nonce_words_found: int
    speaker_similarity: float | None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        """Flat mapping for the audit record."""
        return {
            "passed": self.passed,
            "phrase_cer": round(self.phrase_cer, 4),
            "nonce_words_found": self.nonce_words_found,
            "speaker_similarity": (
                round(self.speaker_similarity, 4) if self.speaker_similarity is not None else None
            ),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class ConsentRecord:
    """A durable, auditable consent record.

    Everything needed to answer an abuse report or a deletion request is on
    this object: who consented, to what, when, how it was verified, and whether
    it is still in force.
    """

    id: str
    voice_id: str
    subject_name: str
    phrase_text: str
    nonce: tuple[str, ...]
    status: ConsentStatus
    verification: ConsentVerification
    consent_audio_path: str
    created_at: datetime
    expires_at: datetime
    revoked_at: datetime | None = None
    jurisdiction: str = "IN"

    @property
    def is_active(self) -> bool:
        """True if the consent is verified, unrevoked and unexpired."""
        if self.status is not ConsentStatus.VERIFIED:
            return False
        if self.revoked_at is not None:
            return False
        return datetime.now(UTC) < self.expires_at

    def effective_status(self) -> ConsentStatus:
        """Status, with expiry and revocation applied."""
        if self.revoked_at is not None:
            return ConsentStatus.REVOKED
        if self.status is ConsentStatus.VERIFIED and datetime.now(UTC) >= self.expires_at:
            return ConsentStatus.EXPIRED
        return self.status


class ConsentVerifier:
    """Checks a consent recording against the issued challenge.

    Args:
        transcriber: ASR for the phrase check. Without one, the phrase cannot be
            verified and :meth:`verify` refuses -- an unverified consent record
            is worse than none, because it looks like evidence.
        speaker_verifier: Speaker verification for the same-speaker check.
            Optional but strongly recommended; without it, anyone can read the
            challenge on the subject's behalf.
        max_phrase_cer: Character error rate above which the phrase is treated
            as not spoken. The default tolerates ASR error and reading
            disfluency while still requiring the phrase.
        min_speaker_similarity: Cosine similarity below which the recordings are
            treated as different speakers. Calibrate against your own verifier's
            score distribution before trusting the default.
    """

    def __init__(
        self,
        *,
        transcriber: Transcriber | None = None,
        speaker_verifier: SpeakerVerifier | None = None,
        max_phrase_cer: float = _MAX_PHRASE_CER,
        min_speaker_similarity: float = _MIN_SPEAKER_SIMILARITY,
    ) -> None:
        self._transcriber = transcriber
        self._speaker_verifier = speaker_verifier
        self._max_phrase_cer = max_phrase_cer
        self._min_similarity = min_speaker_similarity

    def verify(
        self,
        phrase: ConsentPhrase,
        consent_audio: Audio,
        enrolment_audio: Audio,
        *,
        max_challenge_age: timedelta = timedelta(hours=1),
    ) -> ConsentVerification:
        """Verify a consent recording.

        Args:
            phrase: The challenge that was issued.
            consent_audio: Recording of the subject reading the challenge.
            enrolment_audio: The reference sample being enrolled.
            max_challenge_age: How long a challenge stays valid.

        Returns:
            The verification result, including the reasons for a failure.

        Raises:
            ConsentError: No transcriber is configured, so the phrase cannot be
                checked at all.
        """
        if self._transcriber is None:
            raise ConsentError(
                "consent verification requires a transcriber; refusing to "
                "record an unverified consent"
            )

        reasons: list[str] = []
        if not phrase.is_fresh(max_age=max_challenge_age):
            reasons.append("challenge expired before the recording was submitted")

        recognised = self._transcriber.transcribe(consent_audio)
        cer = character_error_rate(phrase.text, recognised).rate
        if cer > self._max_phrase_cer:
            reasons.append(f"spoken phrase does not match the challenge (cer {cer:.2f})")

        found = sum(1 for word in phrase.nonce if word in recognised)
        if found < len(phrase.nonce):
            reasons.append(f"only {found} of {len(phrase.nonce)} confirmation words were spoken")

        similarity: float | None = None
        if self._speaker_verifier is not None:
            similarity = self._speaker_verifier.similarity(consent_audio, enrolment_audio)
            if similarity < self._min_similarity:
                reasons.append(
                    "consent recording and enrolment sample are not the same "
                    f"speaker (similarity {similarity:.2f})"
                )
        else:
            reasons.append("no speaker verifier configured; same-speaker check skipped")

        verification = ConsentVerification(
            passed=not reasons,
            phrase_cer=cer,
            nonce_words_found=found,
            speaker_similarity=similarity,
            reasons=tuple(reasons),
        )
        log.info(
            "consent verification",
            subject=phrase.subject_name,
            **verification.as_dict(),
        )
        return verification
