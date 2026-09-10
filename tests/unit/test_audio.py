"""Audio IO, resampling, loudness and quality gates."""

from __future__ import annotations

import numpy as np
import pytest

from mlvoice.audio.io import (
    Audio,
    encode_wav,
    load_audio,
    resample,
    save_audio,
    to_pcm16,
    wav_header,
)
from mlvoice.audio.loudness import (
    BROADCAST_TARGET_LUFS,
    LoudnessMethod,
    measure_loudness,
    normalize_loudness,
    peak_dbfs,
)
from mlvoice.audio.quality import QualityGate, measure_quality
from mlvoice.audio.vad import EnergyVad, trim_silence
from mlvoice.errors import AudioError
from tests.conftest import SpeechFactory


class TestAudioContainer:
    def test_duration(self, clip: Audio) -> None:
        assert clip.duration_seconds == pytest.approx(clip.num_samples / clip.sample_rate)

    def test_rejects_multichannel(self) -> None:
        with pytest.raises(AudioError):
            Audio(samples=np.zeros((100, 2), dtype=np.float32), sample_rate=24_000)

    def test_rejects_bad_sample_rate(self) -> None:
        with pytest.raises(AudioError):
            Audio(samples=np.zeros(100, dtype=np.float32), sample_rate=0)

    def test_slice_clamps_to_bounds(self, clip: Audio) -> None:
        assert clip.slice_seconds(-5, 1000).num_samples == clip.num_samples

    def test_slice_of_inverted_range_is_empty(self, clip: Audio) -> None:
        assert clip.slice_seconds(2.0, 1.0).num_samples == 0

    def test_with_samples_keeps_the_rate(self, clip: Audio) -> None:
        replaced = clip.with_samples(np.zeros(10))
        assert replaced.sample_rate == clip.sample_rate
        assert replaced.samples.dtype == np.float32


class TestIO:
    def test_wav_round_trip(self, clip: Audio) -> None:
        decoded = load_audio(encode_wav(clip))
        assert decoded.sample_rate == clip.sample_rate
        assert decoded.num_samples == clip.num_samples
        assert np.max(np.abs(decoded.samples - clip.samples)) < 1e-3

    def test_save_and_load_from_path(self, clip: Audio, tmp_path) -> None:
        path = save_audio(clip, tmp_path / "nested" / "a.wav")
        assert path.exists()
        assert load_audio(path).num_samples == clip.num_samples

    def test_undecodable_input_raises(self) -> None:
        with pytest.raises(AudioError):
            load_audio(b"not audio at all")

    def test_stereo_is_averaged_to_mono(self, tmp_path) -> None:
        import soundfile as sf

        stereo = np.stack([np.ones(1000), -np.ones(1000)], axis=1).astype(np.float32)
        sf.write(str(tmp_path / "s.wav"), stereo, 24_000)
        # 16-bit PCM is asymmetric (+1.0 -> 32767, -1.0 -> -32768), so the
        # average of the two channels lands one LSB below zero, not exactly on it.
        assert np.allclose(load_audio(tmp_path / "s.wav").samples, 0.0, atol=1e-4)

    def test_pcm16_length(self, clip: Audio) -> None:
        assert len(to_pcm16(clip)) == clip.num_samples * 2

    def test_streaming_header_is_44_bytes(self) -> None:
        assert len(wav_header(24_000)) == 44

    def test_encode_clips_out_of_range_samples(self) -> None:
        loud = Audio(samples=np.full(1000, 4.0, dtype=np.float32), sample_rate=24_000)
        assert np.max(np.abs(load_audio(encode_wav(loud)).samples)) <= 1.0


class TestResampling:
    def test_same_rate_is_a_no_op(self, clip: Audio) -> None:
        assert resample(clip, clip.sample_rate) is clip

    def test_downsample_halves_the_length(self, clip: Audio) -> None:
        out = resample(clip, clip.sample_rate // 2)
        assert out.sample_rate == clip.sample_rate // 2
        assert abs(out.num_samples - clip.num_samples // 2) <= 2

    def test_tone_survives_a_round_trip(self) -> None:
        sr = 48_000
        t = np.arange(sr, dtype=np.float32) / sr
        tone = Audio(samples=np.sin(2 * np.pi * 440 * t).astype(np.float32), sample_rate=sr)
        back = resample(resample(tone, 24_000), sr)
        overlap = min(back.num_samples, tone.num_samples)
        middle = slice(overlap // 4, 3 * overlap // 4)
        assert np.corrcoef(back.samples[middle], tone.samples[middle])[0, 1] > 0.99

    def test_negative_target_rate_raises(self, clip: Audio) -> None:
        with pytest.raises(AudioError):
            resample(clip, -1)


class TestLoudness:
    def test_bs1770_is_used_when_available(self, clip: Audio) -> None:
        assert measure_loudness(clip).method is LoudnessMethod.ITU_BS1770

    def test_short_clips_fall_back_to_rms(self) -> None:
        short = Audio(samples=np.full(2_000, 0.1, dtype=np.float32), sample_rate=24_000)
        measurement = measure_loudness(short)
        assert measurement.method is LoudnessMethod.RMS_DBFS
        assert measurement.is_true_lufs is False

    def test_normalisation_hits_the_target(self, clip: Audio) -> None:
        levelled, _ = normalize_loudness(clip, target_lufs=BROADCAST_TARGET_LUFS)
        assert measure_loudness(levelled).value == pytest.approx(BROADCAST_TARGET_LUFS, abs=0.5)

    def test_peak_ceiling_is_respected(self, speech: SpeechFactory) -> None:
        quiet = speech(words=10)
        levelled, _ = normalize_loudness(quiet, target_lufs=0.0, max_peak_dbfs=-3.0)
        assert peak_dbfs(levelled) <= -3.0 + 0.01

    def test_gain_is_capped(self) -> None:
        very_quiet = Audio(
            samples=(np.random.default_rng(0).normal(0, 1e-6, 48_000)).astype(np.float32),
            sample_rate=24_000,
        )
        levelled, _ = normalize_loudness(very_quiet, target_lufs=-16.0, max_gain_db=6.0)
        assert peak_dbfs(levelled) < -60.0

    def test_silence_cannot_be_normalised(self) -> None:
        silence = Audio(samples=np.zeros(48_000, dtype=np.float32), sample_rate=24_000)
        with pytest.raises(AudioError):
            normalize_loudness(silence)

    def test_peak_of_silence_is_negative_infinity(self) -> None:
        silence = Audio(samples=np.zeros(100, dtype=np.float32), sample_rate=24_000)
        assert peak_dbfs(silence) == float("-inf")


class TestQualityGates:
    def test_clean_speech_passes(self, clip: Audio) -> None:
        passed, reasons = QualityGate().evaluate(measure_quality(clip))
        assert passed, reasons

    def test_clipping_is_detected(self, clip: Audio) -> None:
        clipped = clip.with_samples(np.clip(clip.samples * 30, -1, 1))
        passed, reasons = QualityGate().evaluate(measure_quality(clipped))
        assert not passed
        assert any("clipped" in reason for reason in reasons)

    def test_noise_fails_the_snr_gate(self) -> None:
        noise = Audio(
            samples=np.random.default_rng(1).normal(0, 0.2, 48_000).astype(np.float32),
            sample_rate=24_000,
        )
        passed, reasons = QualityGate().evaluate(measure_quality(noise))
        assert not passed
        assert any("snr" in reason for reason in reasons)

    def test_dc_offset_is_detected(self, clip: Audio) -> None:
        offset = clip.with_samples(clip.samples + 0.2)
        passed, reasons = QualityGate().evaluate(measure_quality(offset))
        assert not passed
        assert any("dc offset" in reason for reason in reasons)

    def test_too_short_is_rejected(self, speech: SpeechFactory) -> None:
        tiny = speech(words=1, word_seconds=0.05, lead_seconds=0.02, gap_seconds=0.01)
        passed, reasons = QualityGate().evaluate(measure_quality(tiny))
        assert not passed
        assert any("too short" in reason for reason in reasons)

    def test_mostly_silence_is_rejected(self, speech: SpeechFactory) -> None:
        sparse = speech(words=1, word_seconds=0.2, gap_seconds=0.1, lead_seconds=3.0)
        passed, reasons = QualityGate().evaluate(measure_quality(sparse))
        assert not passed
        assert any("silence" in reason for reason in reasons)

    def test_report_is_serialisable(self, clip: Audio) -> None:
        payload = measure_quality(clip).as_dict()
        assert set(payload) == {
            "duration_seconds",
            "peak_dbfs",
            "clipping_ratio",
            "dc_offset",
            "silence_ratio",
            "estimated_snr_db",
        }
        assert all(isinstance(v, float) for v in payload.values())

    def test_snr_estimate_survives_continuous_speech(self, speech: SpeechFactory) -> None:
        """A clip with few pauses must not be scored as if it were all noise."""
        continuous = speech(words=12, word_seconds=0.4, gap_seconds=0.02)
        assert measure_quality(continuous).estimated_snr_db > 20.0


class TestVad:
    def test_speech_span_is_located(self, speech: SpeechFactory) -> None:
        audio = speech(words=3, lead_seconds=1.0)
        segments = EnergyVad().detect(audio)
        assert segments
        assert segments[0].start > 0.5
        assert segments[-1].end < audio.duration_seconds - 0.4

    def test_trimming_shortens_but_keeps_speech(self, speech: SpeechFactory) -> None:
        audio = speech(words=4, lead_seconds=1.5)
        trimmed = trim_silence(audio)
        assert trimmed.duration_seconds < audio.duration_seconds
        assert trimmed.duration_seconds > 1.0

    def test_interior_pauses_are_preserved(self, speech: SpeechFactory) -> None:
        audio = speech(words=6, gap_seconds=0.3, lead_seconds=0.5)
        trimmed = trim_silence(audio)
        expected = 6 * 0.32 + 5 * 0.3
        assert trimmed.duration_seconds > expected * 0.9

    def test_silence_is_returned_unchanged(self) -> None:
        silence = Audio(samples=np.zeros(48_000, dtype=np.float32), sample_rate=24_000)
        assert trim_silence(silence).num_samples == silence.num_samples

    def test_segments_report_duration(self, clip: Audio) -> None:
        for segment in EnergyVad().detect(clip):
            assert segment.duration == pytest.approx(segment.end - segment.start)

    def test_empty_audio_yields_no_segments(self) -> None:
        tiny = Audio(samples=np.zeros(5, dtype=np.float32), sample_rate=24_000)
        assert EnergyVad().detect(tiny) == []
