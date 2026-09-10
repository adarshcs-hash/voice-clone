"""Watermarking and moderation."""

from __future__ import annotations

import numpy as np
import pytest

from mlvoice.audio.io import Audio, encode_wav, load_audio
from mlvoice.errors import ValidationError
from mlvoice.safety.moderation import (
    ModerationDecision,
    NameBlocklist,
    PatternModerator,
    Severity,
)
from mlvoice.safety.watermark import PAYLOAD_BITS, SpreadSpectrumWatermarker, payload_for
from tests.conftest import SpeechFactory

KEY = b"test-watermark-key"


@pytest.fixture
def marker() -> SpreadSpectrumWatermarker:
    return SpreadSpectrumWatermarker(KEY)


@pytest.fixture
def long_clip(speech: SpeechFactory) -> Audio:
    return speech(words=14)


class TestWatermarkPayload:
    def test_payload_is_stable(self) -> None:
        assert payload_for("req", "voice") == payload_for("req", "voice")

    def test_payload_depends_on_every_part(self) -> None:
        assert payload_for("req", "voice-a") != payload_for("req", "voice-b")

    def test_payload_fits_in_the_field(self) -> None:
        assert 0 <= payload_for("a", "b") < 2**PAYLOAD_BITS


class TestWatermarkEmbedAndDetect:
    def test_payload_round_trips(self, marker: SpreadSpectrumWatermarker, long_clip: Audio) -> None:
        payload = payload_for("req-1", "voice-1")
        detection = marker.detect(marker.embed(long_clip, payload))
        assert detection.detected
        assert detection.payload == payload

    def test_clean_audio_is_not_detected(
        self, marker: SpreadSpectrumWatermarker, long_clip: Audio
    ) -> None:
        assert not marker.detect(long_clip).detected

    def test_a_different_key_does_not_detect(self, long_clip: Audio) -> None:
        marked = SpreadSpectrumWatermarker(KEY).embed(long_clip, 4242)
        assert not SpreadSpectrumWatermarker(b"other-key").detect(marked).detected

    def test_survives_a_gain_change(
        self, marker: SpreadSpectrumWatermarker, long_clip: Audio
    ) -> None:
        marked = marker.embed(long_clip, 12345)
        quieter = marked.with_samples(marked.samples * 0.25)
        assert marker.detect(quieter).payload == 12345

    def test_survives_a_wav_round_trip(
        self, marker: SpreadSpectrumWatermarker, long_clip: Audio
    ) -> None:
        marked = marker.embed(long_clip, 999)
        assert marker.detect(load_audio(encode_wav(marked))).payload == 999

    def test_survives_added_noise(
        self, marker: SpreadSpectrumWatermarker, long_clip: Audio
    ) -> None:
        marked = marker.embed(long_clip, 777)
        rng = np.random.default_rng(0)
        noisy = marked.with_samples(
            (marked.samples + rng.normal(0, 0.004, marked.num_samples)).astype(np.float32)
        )
        assert marker.detect(noisy).payload == 777

    def test_embedding_is_quiet(self, marker: SpreadSpectrumWatermarker, long_clip: Audio) -> None:
        marked = marker.embed(long_clip, 5)
        difference = float(np.std(marked.samples - long_clip.samples))
        assert 20 * np.log10(difference / float(np.std(long_clip.samples))) < -20.0

    def test_output_does_not_clip(
        self, marker: SpreadSpectrumWatermarker, long_clip: Audio
    ) -> None:
        assert np.max(np.abs(marker.embed(long_clip, 1).samples)) <= 1.0

    def test_too_short_to_mark(self, marker: SpreadSpectrumWatermarker) -> None:
        tiny = Audio(samples=np.zeros(500, dtype=np.float32), sample_rate=24_000)
        with pytest.raises(ValidationError, match="too short"):
            marker.embed(tiny, 1)

    def test_detection_on_a_short_clip_is_negative(self, marker: SpreadSpectrumWatermarker) -> None:
        tiny = Audio(samples=np.zeros(500, dtype=np.float32), sample_rate=24_000)
        assert marker.detect(tiny).detected is False

    def test_oversized_payload_is_refused(
        self, marker: SpreadSpectrumWatermarker, long_clip: Audio
    ) -> None:
        with pytest.raises(ValidationError, match="32 bits"):
            marker.embed(long_clip, 2**PAYLOAD_BITS)

    def test_empty_key_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            SpreadSpectrumWatermarker(b"")

    def test_detection_is_serialisable(
        self, marker: SpreadSpectrumWatermarker, long_clip: Audio
    ) -> None:
        payload = marker.detect(marker.embed(long_clip, 3)).as_dict()
        assert set(payload) == {"detected", "payload", "confidence"}


class TestModeration:
    @pytest.fixture
    def moderator(self) -> PatternModerator:
        return PatternModerator()

    @pytest.mark.parametrize(
        "text",
        [
            "നന്ദി, നാളെ കാണാം",
            "കേരളത്തിലെ കാലാവസ്ഥ ഇന്ന് നല്ലതാണ്",
            "ഈ ഓഡിയോ ബുക്കിൽ പോലീസ് കഥ ഉണ്ട്",
            "ബാങ്കിൽ പോയി വരാം",
        ],
    )
    def test_ordinary_text_is_allowed(self, moderator: PatternModerator, text: str) -> None:
        assert moderator.review(text).severity is Severity.ALLOW

    @pytest.mark.parametrize(
        ("text", "rule"),
        [
            ("എന്റെ OTP ഉടനെ പറയണം", "otp_solicitation"),
            ("Share the OTP now", "otp_solicitation"),
            ("Please share your password immediately", "credential_solicitation"),
            ("അടിയന്തരമായി പണം അയക്കണം", "urgent_money_transfer"),
            (
                "മോൻ അപകടത്തിൽ പെട്ടു, ആശുപത്രിയിൽ ആണ്, ഉടനെ പണം അയക്കൂ",
                "relative_in_trouble_scam",
            ),
        ],
    )
    def test_fraud_templates_are_blocked(
        self, moderator: PatternModerator, text: str, rule: str
    ) -> None:
        decision = moderator.review(text)
        assert decision.blocked
        assert rule in decision.rules

    @pytest.mark.parametrize(
        ("text", "rule"),
        [
            ("നിങ്ങളുടെ account number തരാമോ", "account_details_request"),
            ("ലോട്ടറി അടിച്ചു, സമ്മാനം claim ചെയ്യൂ", "prize_lottery_scam"),
            ("ഞാൻ ആണ് മുഖ്യമന്ത്രി", "impersonation_claim"),
        ],
    )
    def test_suspicious_text_is_flagged_not_blocked(
        self, moderator: PatternModerator, text: str, rule: str
    ) -> None:
        decision = moderator.review(text)
        assert decision.severity is Severity.FLAG
        assert not decision.blocked
        assert rule in decision.rules

    def test_matching_is_order_independent(self, moderator: PatternModerator) -> None:
        assert moderator.review("OTP അയക്കൂ").blocked
        assert moderator.review("അയക്കൂ എന്റെ OTP").blocked

    def test_extra_block_patterns_are_applied(self) -> None:
        moderator = PatternModerator(extra_block_patterns=[("custom", r"നിരോധിത")])
        decision = moderator.review("ഇത് നിരോധിത വാക്ക്")
        assert decision.blocked
        assert "custom" in decision.rules

    def test_decision_is_serialisable(self, moderator: PatternModerator) -> None:
        payload = moderator.review("OTP അയക്കൂ").as_dict()
        assert payload["severity"] == "block"
        assert payload["rules"]

    def test_allow_decision_has_no_rules(self) -> None:
        assert ModerationDecision(Severity.ALLOW).rules == ()


class TestNameBlocklist:
    def test_exact_and_fuzzy_matching(self) -> None:
        blocklist = NameBlocklist(["Pinarayi Vijayan", "Mohanlal"])
        assert blocklist.is_blocked("Mohanlal")
        assert blocklist.is_blocked("MOHANLAL")
        assert blocklist.is_blocked("mohan lal")
        assert blocklist.is_blocked("CM Pinarayi Vijayan (test)")

    def test_unrelated_names_pass(self) -> None:
        assert not NameBlocklist(["Mohanlal"]).is_blocked("Rajan Nair")

    def test_empty_name_is_not_blocked(self) -> None:
        assert not NameBlocklist(["Mohanlal"]).is_blocked("   ")

    def test_empty_blocklist_blocks_nothing(self) -> None:
        assert not NameBlocklist().is_blocked("anyone")
        assert len(NameBlocklist()) == 0

    def test_loads_from_a_file(self, tmp_path) -> None:
        path = tmp_path / "blocked.txt"
        path.write_text("# comment\nMohanlal\n\nMammootty\n", encoding="utf-8")
        blocklist = NameBlocklist.from_file(path)
        assert len(blocklist) == 2
        assert blocklist.is_blocked("mammootty")

    def test_missing_file_is_tolerated(self, tmp_path) -> None:
        assert len(NameBlocklist.from_file(tmp_path / "absent.txt")) == 0

    def test_a_directory_is_tolerated(self, tmp_path) -> None:
        """Regression: a misconfigured path pointing at a directory raised
        IsADirectoryError and killed application startup."""
        assert len(NameBlocklist.from_file(tmp_path)) == 0
        assert len(NameBlocklist.from_file(".")) == 0
