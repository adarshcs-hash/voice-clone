"""Consent, persistence and enrolment.

These tests exist to pin the safety invariants: consent cannot be forged,
replayed or skipped; revocation cascades; deletion removes the audio.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mlvoice.audio.io import Audio, encode_wav
from mlvoice.config import Settings
from mlvoice.errors import (
    AudioError,
    ConsentError,
    ModerationError,
    ValidationError,
    VoiceNotFoundError,
)
from mlvoice.safety.moderation import NameBlocklist
from mlvoice.voices.consent import (
    ConsentPhrase,
    ConsentRecord,
    ConsentStatus,
    ConsentVerification,
    ConsentVerifier,
    issue_phrase,
)
from mlvoice.voices.enrollment import EnrollmentRequest, EnrollmentService
from mlvoice.voices.store import Voice, VoiceStatus, VoiceStore
from tests.conftest import FakeSpeakerVerifier, FakeTranscriber, SpeechFactory


class TestConsentPhrase:
    def test_phrase_contains_name_date_and_nonce(self) -> None:
        phrase = issue_phrase("രാജൻ നായർ")
        assert "രാജൻ നായർ" in phrase.text
        assert str(datetime.now(UTC).year) in phrase.text
        for word in phrase.nonce:
            assert word in phrase.text

    def test_nonce_varies_between_challenges(self) -> None:
        nonces = {issue_phrase("രാജൻ").nonce for _ in range(20)}
        assert len(nonces) > 1, "a fixed nonce would allow a replayed recording"

    def test_freshness(self) -> None:
        phrase = issue_phrase("രാജൻ")
        assert phrase.is_fresh()
        stale = ConsentPhrase(
            text=phrase.text,
            nonce=phrase.nonce,
            issued_at=datetime.now(UTC) - timedelta(hours=3),
            subject_name="രാജൻ",
        )
        assert not stale.is_fresh()

    def test_empty_name_is_refused(self) -> None:
        with pytest.raises(ConsentError):
            issue_phrase("  ")


class TestConsentVerification:
    def test_matching_recording_passes(
        self,
        clip: Audio,
        transcriber: FakeTranscriber,
        speaker_verifier: FakeSpeakerVerifier,
    ) -> None:
        phrase = issue_phrase("രാജൻ")
        transcriber.text = phrase.text
        verification = ConsentVerifier(
            transcriber=transcriber, speaker_verifier=speaker_verifier
        ).verify(phrase, clip, clip)
        assert verification.passed
        assert verification.phrase_cer == pytest.approx(0.0)
        assert verification.nonce_words_found == len(phrase.nonce)

    def test_wrong_phrase_fails(
        self,
        clip: Audio,
        transcriber: FakeTranscriber,
        speaker_verifier: FakeSpeakerVerifier,
    ) -> None:
        phrase = issue_phrase("രാജൻ")
        transcriber.text = "തികച്ചും വേറൊരു വാക്യം"
        verification = ConsentVerifier(
            transcriber=transcriber, speaker_verifier=speaker_verifier
        ).verify(phrase, clip, clip)
        assert not verification.passed

    def test_missing_nonce_words_fail(
        self,
        clip: Audio,
        transcriber: FakeTranscriber,
        speaker_verifier: FakeSpeakerVerifier,
    ) -> None:
        phrase = issue_phrase("രാജൻ")
        # Same sentence minus the confirmation words: low CER, missing nonce.
        transcriber.text = phrase.text.split("സ്ഥിരീകരണ")[0]
        verification = ConsentVerifier(
            transcriber=transcriber,
            speaker_verifier=speaker_verifier,
            max_phrase_cer=0.9,
        ).verify(phrase, clip, clip)
        assert not verification.passed
        assert any("confirmation words" in reason for reason in verification.reasons)

    def test_different_speaker_fails(self, clip: Audio, transcriber: FakeTranscriber) -> None:
        phrase = issue_phrase("രാജൻ")
        transcriber.text = phrase.text
        verification = ConsentVerifier(
            transcriber=transcriber, speaker_verifier=FakeSpeakerVerifier(score=0.2)
        ).verify(phrase, clip, clip)
        assert not verification.passed
        assert any("same speaker" in reason for reason in verification.reasons)

    def test_stale_challenge_fails(
        self,
        clip: Audio,
        transcriber: FakeTranscriber,
        speaker_verifier: FakeSpeakerVerifier,
    ) -> None:
        phrase = issue_phrase("രാജൻ")
        transcriber.text = phrase.text
        stale = ConsentPhrase(
            text=phrase.text,
            nonce=phrase.nonce,
            issued_at=datetime.now(UTC) - timedelta(hours=5),
            subject_name="രാജൻ",
        )
        verification = ConsentVerifier(
            transcriber=transcriber, speaker_verifier=speaker_verifier
        ).verify(stale, clip, clip)
        assert not verification.passed
        assert any("expired" in reason for reason in verification.reasons)

    def test_missing_speaker_verifier_is_recorded_as_a_gap(
        self, clip: Audio, transcriber: FakeTranscriber
    ) -> None:
        phrase = issue_phrase("രാജൻ")
        transcriber.text = phrase.text
        verification = ConsentVerifier(transcriber=transcriber).verify(phrase, clip, clip)
        assert not verification.passed
        assert any("speaker verifier" in reason for reason in verification.reasons)

    def test_no_transcriber_is_refused_outright(self, clip: Audio) -> None:
        with pytest.raises(ConsentError, match="transcriber"):
            ConsentVerifier().verify(issue_phrase("രാജൻ"), clip, clip)


class TestConsentRecord:
    def _record(self, **overrides: object) -> ConsentRecord:
        now = datetime.now(UTC)
        payload: dict[str, object] = {
            "id": "c1",
            "voice_id": "v1",
            "subject_name": "രാജൻ",
            "phrase_text": "ഞാൻ രാജൻ",
            "nonce": ("മഴ",),
            "status": ConsentStatus.VERIFIED,
            "verification": ConsentVerification(
                passed=True, phrase_cer=0.0, nonce_words_found=1, speaker_similarity=0.9
            ),
            "consent_audio_path": "/tmp/c.wav",
            "created_at": now,
            "expires_at": now + timedelta(days=365),
        }
        payload.update(overrides)
        return ConsentRecord(**payload)  # type: ignore[arg-type]

    def test_verified_record_is_active(self) -> None:
        assert self._record().is_active

    def test_revoked_record_is_inactive(self) -> None:
        record = self._record(revoked_at=datetime.now(UTC))
        assert not record.is_active
        assert record.effective_status() is ConsentStatus.REVOKED

    def test_expired_record_is_inactive(self) -> None:
        record = self._record(expires_at=datetime.now(UTC) - timedelta(days=1))
        assert not record.is_active
        assert record.effective_status() is ConsentStatus.EXPIRED

    def test_pending_record_is_inactive(self) -> None:
        assert not self._record(status=ConsentStatus.PENDING).is_active


class TestVoiceStore:
    def _voice(self, **overrides: object) -> Voice:
        now = datetime.now(UTC)
        payload: dict[str, object] = {
            "id": "v1",
            "name": "Test voice",
            "owner_id": "acct1",
            "reference_audio_path": "/tmp/ref.wav",
            "reference_text": "ടെസ്റ്റ്",
            "reference_duration_seconds": 6.0,
            "sample_rate": 24_000,
            "status": VoiceStatus.ACTIVE,
            "consent_id": None,
            "created_at": now,
            "updated_at": now,
        }
        payload.update(overrides)
        return Voice(**payload)  # type: ignore[arg-type]

    def test_create_and_read(self, store: VoiceStore) -> None:
        created = store.create_voice(self._voice())
        assert store.get_voice(created.id).name == "Test voice"

    def test_timezone_survives_the_round_trip(self, store: VoiceStore) -> None:
        store.create_voice(self._voice())
        assert store.get_voice("v1").created_at.tzinfo is not None

    def test_owner_scoping_hides_other_tenants(self, store: VoiceStore) -> None:
        store.create_voice(self._voice())
        with pytest.raises(VoiceNotFoundError):
            store.get_voice("v1", owner_id="someone-else")

    def test_listing_is_scoped_and_newest_first(self, store: VoiceStore) -> None:
        now = datetime.now(UTC)
        store.create_voice(self._voice(id="a", created_at=now - timedelta(minutes=5)))
        store.create_voice(self._voice(id="b", created_at=now))
        store.create_voice(self._voice(id="c", owner_id="other"))
        assert [v.id for v in store.list_voices(owner_id="acct1")] == ["b", "a"]

    def test_pagination(self, store: VoiceStore) -> None:
        for index in range(5):
            store.create_voice(
                self._voice(id=f"v{index}", created_at=datetime.now(UTC) - timedelta(minutes=index))
            )
        assert len(store.list_voices(owner_id="acct1", limit=2)) == 2
        assert len(store.list_voices(owner_id="acct1", limit=2, offset=4)) == 1

    def test_status_change(self, store: VoiceStore) -> None:
        store.create_voice(self._voice())
        assert store.set_voice_status("v1", VoiceStatus.DISABLED).status is VoiceStatus.DISABLED
        assert not store.get_voice("v1").is_usable

    def test_missing_voice_raises(self, store: VoiceStore) -> None:
        with pytest.raises(VoiceNotFoundError):
            store.get_voice("absent")
        with pytest.raises(VoiceNotFoundError):
            store.set_voice_status("absent", VoiceStatus.DISABLED)
        with pytest.raises(VoiceNotFoundError):
            store.delete_voice("absent")

    def test_deletion_removes_the_audio(self, store: VoiceStore, tmp_path: Path) -> None:
        reference = tmp_path / "ref.wav"
        reference.write_bytes(b"placeholder")
        store.create_voice(self._voice(reference_audio_path=str(reference)))
        store.delete_voice("v1", owner_id="acct1")
        assert not reference.exists()
        with pytest.raises(VoiceNotFoundError):
            store.get_voice("v1")

    def test_deletion_tolerates_missing_files(self, store: VoiceStore) -> None:
        store.create_voice(self._voice(reference_audio_path="/nonexistent/ref.wav"))
        store.delete_voice("v1")

    def test_missing_consent_raises(self, store: VoiceStore) -> None:
        with pytest.raises(VoiceNotFoundError):
            store.get_consent("absent")


class TestEnrollment:
    @pytest.fixture
    def service(
        self,
        settings: Settings,
        store: VoiceStore,
        verifier: ConsentVerifier,
    ) -> EnrollmentService:
        return EnrollmentService(
            settings, store, verifier=verifier, blocklist=NameBlocklist(["Mohanlal"])
        )

    @pytest.fixture
    def audio_bytes(self, speech: SpeechFactory) -> bytes:
        return encode_wav(speech(words=14, seed=5))

    def _request(self, token: str, audio: bytes, **overrides: object) -> EnrollmentRequest:
        payload: dict[str, object] = {
            "name": "Rajan voice",
            "owner_id": "acct1",
            "reference_audio": audio,
            "reference_text": "ഇത് എന്റെ ശബ്ദ സാമ്പിൾ ആണ്",
            "consent_token": token,
            "consent_audio": audio,
        }
        payload.update(overrides)
        return EnrollmentRequest(**payload)  # type: ignore[arg-type]

    def test_successful_enrolment(
        self,
        service: EnrollmentService,
        transcriber: FakeTranscriber,
        audio_bytes: bytes,
    ) -> None:
        issued = service.issue_challenge("രാജൻ നായർ")
        transcriber.text = issued.phrase.text
        voice = service.enroll(self._request(issued.token, audio_bytes))
        assert voice.status is VoiceStatus.ACTIVE
        assert voice.consent_id is not None
        assert Path(voice.reference_audio_path).exists()
        assert voice.quality

    def test_reference_transcript_is_required(
        self,
        service: EnrollmentService,
        transcriber: FakeTranscriber,
        audio_bytes: bytes,
    ) -> None:
        issued = service.issue_challenge("രാജൻ")
        transcriber.text = issued.phrase.text
        with pytest.raises(ValidationError, match="reference transcript"):
            service.enroll(self._request(issued.token, audio_bytes, reference_text="  "))

    def test_forged_token_is_refused(self, service: EnrollmentService, audio_bytes: bytes) -> None:
        issued = service.issue_challenge("രാജൻ")
        tampered = issued.token[:-6] + "AAAAAA"
        with pytest.raises(ConsentError, match="signature"):
            service.enroll(self._request(tampered, audio_bytes))

    def test_malformed_token_is_refused(
        self, service: EnrollmentService, audio_bytes: bytes
    ) -> None:
        with pytest.raises(ConsentError, match="malformed"):
            service.enroll(self._request("nonsense", audio_bytes))

    def test_token_from_another_deployment_is_refused(
        self,
        settings: Settings,
        store: VoiceStore,
        verifier: ConsentVerifier,
        audio_bytes: bytes,
    ) -> None:
        other = EnrollmentService(
            settings.model_copy(update={"consent_signing_key": settings.consent_signing_key}),
            store,
            verifier=verifier,
        )
        foreign_settings = Settings(
            _env_file=None,
            consent_signing_key="a-completely-different-key",
            database_url=settings.database_url,
            voice_storage_dir=settings.voice_storage_dir,
        )
        foreign = EnrollmentService(foreign_settings, store, verifier=verifier)
        issued = foreign.issue_challenge("രാജൻ")
        with pytest.raises(ConsentError, match="signature"):
            other.enroll(self._request(issued.token, audio_bytes))

    def test_blocked_name_is_refused(
        self,
        service: EnrollmentService,
        transcriber: FakeTranscriber,
        audio_bytes: bytes,
    ) -> None:
        issued = service.issue_challenge("Mohanlal")
        transcriber.text = issued.phrase.text
        with pytest.raises(ModerationError):
            service.enroll(self._request(issued.token, audio_bytes, name="Mohanlal clone"))

    def test_failed_consent_blocks_enrolment(
        self,
        service: EnrollmentService,
        transcriber: FakeTranscriber,
        audio_bytes: bytes,
    ) -> None:
        issued = service.issue_challenge("രാജൻ")
        transcriber.text = "വേറൊരു വാക്യം"
        with pytest.raises(ConsentError, match="verification failed"):
            service.enroll(self._request(issued.token, audio_bytes))

    def test_poor_reference_audio_is_refused(
        self,
        service: EnrollmentService,
        transcriber: FakeTranscriber,
    ) -> None:
        import numpy as np

        issued = service.issue_challenge("രാജൻ")
        transcriber.text = issued.phrase.text
        noise = Audio(
            samples=np.random.default_rng(0).normal(0, 0.2, 24_000 * 6).astype(np.float32),
            sample_rate=24_000,
        )
        with pytest.raises(AudioError, match="quality"):
            service.enroll(self._request(issued.token, encode_wav(noise)))

    def test_consent_required_without_a_verifier_is_refused(
        self, settings: Settings, store: VoiceStore, audio_bytes: bytes
    ) -> None:
        service = EnrollmentService(settings, store, verifier=None)
        issued = service.issue_challenge("രാജൻ")
        with pytest.raises(ConsentError, match="no verifier"):
            service.enroll(self._request(issued.token, audio_bytes))

    def test_revocation_cascades_to_voices(
        self,
        service: EnrollmentService,
        store: VoiceStore,
        transcriber: FakeTranscriber,
        audio_bytes: bytes,
    ) -> None:
        issued = service.issue_challenge("രാജൻ")
        transcriber.text = issued.phrase.text
        voice = service.enroll(self._request(issued.token, audio_bytes))
        assert voice.consent_id is not None

        store.revoke_consent(voice.consent_id)
        assert store.get_voice(voice.id).status is VoiceStatus.DISABLED
        assert not store.get_consent(voice.consent_id).is_active

    def test_deletion_removes_consent_audio_too(
        self,
        service: EnrollmentService,
        store: VoiceStore,
        transcriber: FakeTranscriber,
        audio_bytes: bytes,
    ) -> None:
        issued = service.issue_challenge("രാജൻ")
        transcriber.text = issued.phrase.text
        voice = service.enroll(self._request(issued.token, audio_bytes))
        assert voice.consent_id is not None
        consent_path = Path(store.get_consent(voice.consent_id).consent_audio_path)
        assert consent_path.exists()

        store.delete_voice(voice.id, owner_id="acct1")
        assert not consent_path.exists()
        assert not Path(voice.reference_audio_path).exists()
