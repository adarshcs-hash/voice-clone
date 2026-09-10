"""Synthesis backend contract and the deterministic stub."""

from __future__ import annotations

import numpy as np
import pytest

from mlvoice.audio.io import Audio
from mlvoice.config import Settings
from mlvoice.errors import ConfigurationError, ValidationError
from mlvoice.text.pipeline import TextPipeline
from mlvoice.tts.base import (
    MAX_REFERENCE_SECONDS,
    ReferencePrompt,
    SynthesisRequest,
)
from mlvoice.tts.dummy import DummySynthesizer
from mlvoice.tts.registry import available_backends, build_synthesizer, register_backend


@pytest.fixture
def pipeline() -> TextPipeline:
    return TextPipeline()


@pytest.fixture
def backend() -> DummySynthesizer:
    synthesizer = DummySynthesizer()
    synthesizer.load()
    return synthesizer


class TestReferencePrompt:
    def test_requires_a_transcript(self, clip: Audio) -> None:
        with pytest.raises(ValidationError, match="transcript"):
            ReferencePrompt(audio=clip, text="   ", voice_id="v1")

    def test_rejects_audio_that_is_too_short(self) -> None:
        short = Audio(samples=np.zeros(2_000, dtype=np.float32), sample_rate=24_000)
        with pytest.raises(ValidationError, match="too short"):
            ReferencePrompt(audio=short, text="ടെസ്റ്റ്", voice_id="v1")

    def test_rejects_audio_that_is_too_long(self) -> None:
        long = Audio(
            samples=np.zeros(int(24_000 * (MAX_REFERENCE_SECONDS + 5)), dtype=np.float32),
            sample_rate=24_000,
        )
        with pytest.raises(ValidationError, match="longer"):
            ReferencePrompt(audio=long, text="ടെസ്റ്റ്", voice_id="v1")

    def test_accepts_a_valid_prompt(self, clip: Audio) -> None:
        prompt = ReferencePrompt(audio=clip, text="ടെസ്റ്റ്", voice_id="v1")
        assert prompt.voice_id == "v1"


class TestSynthesisRequest:
    def test_speed_bounds(self, pipeline: TextPipeline) -> None:
        processed = pipeline.process("നാട്")
        for speed in (0.4, 2.5):
            with pytest.raises(ValidationError, match="speed"):
                SynthesisRequest(text=processed, speed=speed)

    def test_valid_speed(self, pipeline: TextPipeline) -> None:
        SynthesisRequest(text=pipeline.process("നാട്"), speed=1.5)


class TestDummyBackend:
    def test_reports_readiness(self) -> None:
        synthesizer = DummySynthesizer()
        assert not synthesizer.is_ready()
        synthesizer.load()
        assert synthesizer.is_ready()

    def test_load_is_idempotent(self, backend: DummySynthesizer) -> None:
        backend.load()
        assert backend.is_ready()

    def test_synthesis_produces_audio(
        self, backend: DummySynthesizer, pipeline: TextPipeline
    ) -> None:
        result = backend.synthesize(SynthesisRequest(text=pipeline.process("ഞാൻ പോയി.")))
        assert result.audio.duration_seconds > 0.1
        assert result.audio.sample_rate == 24_000

    def test_output_is_deterministic(
        self, backend: DummySynthesizer, pipeline: TextPipeline
    ) -> None:
        processed = pipeline.process("ഞാൻ നാട്ടിൽ പോയി.")
        first = backend.synthesize(SynthesisRequest(text=processed, seed=7)).audio
        second = backend.synthesize(SynthesisRequest(text=processed, seed=7)).audio
        assert np.array_equal(first.samples, second.samples)

    def test_longer_text_produces_longer_audio(
        self, backend: DummySynthesizer, pipeline: TextPipeline
    ) -> None:
        short = backend.synthesize(SynthesisRequest(text=pipeline.process("നാട്")))
        long = backend.synthesize(
            SynthesisRequest(text=pipeline.process("ഞാൻ എന്റെ നാട്ടിലേക്ക് തിരികെ പോയി."))
        )
        assert long.audio.duration_seconds > short.audio.duration_seconds

    def test_speed_shortens_the_output(
        self, backend: DummySynthesizer, pipeline: TextPipeline
    ) -> None:
        processed = pipeline.process("ഞാൻ നാട്ടിൽ പോയി.")
        slow = backend.synthesize(SynthesisRequest(text=processed, speed=0.5))
        fast = backend.synthesize(SynthesisRequest(text=processed, speed=2.0))
        assert fast.audio.duration_seconds < slow.audio.duration_seconds

    def test_different_voices_differ(
        self, backend: DummySynthesizer, pipeline: TextPipeline, clip: Audio
    ) -> None:
        processed = pipeline.process("ഞാൻ പോയി.")
        first = backend.synthesize(
            SynthesisRequest(
                text=processed, prompt=ReferencePrompt(audio=clip, text="a", voice_id="v1")
            )
        ).audio
        second = backend.synthesize(
            SynthesisRequest(
                text=processed, prompt=ReferencePrompt(audio=clip, text="a", voice_id="v2")
            )
        ).audio
        assert not np.array_equal(first.samples, second.samples)


class TestChunkingAndStreaming:
    def test_stream_yields_one_buffer_per_chunk(
        self, backend: DummySynthesizer, pipeline: TextPipeline
    ) -> None:
        processed = pipeline.process(
            "ഒന്നാമത്തെ വാക്യം ഇവിടെ അവസാനിക്കുന്നു. "
            "രണ്ടാമത്തെ വാക്യം ഇവിടെ അവസാനിക്കുന്നു. "
            "മൂന്നാമത്തെ വാക്യം ഇവിടെ അവസാനിക്കുന്നു."
        )
        assert len(list(backend.stream(SynthesisRequest(text=processed)))) == len(processed.chunks)

    def test_pauses_are_inserted_between_chunks(
        self, backend: DummySynthesizer, pipeline: TextPipeline
    ) -> None:
        processed = pipeline.process("ഒന്നാമത്തെ വാക്യം ഇവിടെ അവസാനിക്കുന്നു. രണ്ടാമത്തെ വാക്യം ഇവിടെ അവസാനിക്കുന്നു.")
        buffers = list(backend.stream(SynthesisRequest(text=processed)))
        if len(buffers) > 1:
            tail = buffers[0].samples[-1_000:]
            assert np.max(np.abs(tail)) < 1e-6, "a chunk with a pause must end in silence"

    def test_concatenation_matches_the_stream(
        self, backend: DummySynthesizer, pipeline: TextPipeline
    ) -> None:
        processed = pipeline.process("ഒന്ന്. രണ്ട്.")
        request = SynthesisRequest(text=processed, seed=1)
        streamed = np.concatenate([b.samples for b in backend.stream(request)])
        assert np.array_equal(backend.synthesize(request).audio.samples, streamed)


class TestResultMetrics:
    def test_real_time_factor(self, backend: DummySynthesizer, pipeline: TextPipeline) -> None:
        result = backend.synthesize(SynthesisRequest(text=pipeline.process("ഞാൻ പോയി.")))
        assert result.real_time_factor > 0
        assert result.real_time_factor < 1.0  # the stub is far faster than real time

    def test_metrics_are_log_safe(self, backend: DummySynthesizer, pipeline: TextPipeline) -> None:
        metrics = backend.synthesize(SynthesisRequest(text=pipeline.process("നാട്"))).metrics()
        assert set(metrics) == {
            "audio_seconds",
            "generation_seconds",
            "real_time_factor",
            "chunks",
            "backend",
        }

    def test_voice_id_is_carried_through(
        self, backend: DummySynthesizer, pipeline: TextPipeline, clip: Audio
    ) -> None:
        result = backend.synthesize(
            SynthesisRequest(
                text=pipeline.process("നാട്"),
                prompt=ReferencePrompt(audio=clip, text="a", voice_id="v9"),
            )
        )
        assert result.voice_id == "v9"

    def test_info_is_serialisable(self, backend: DummySynthesizer) -> None:
        payload = backend.info.as_dict()
        assert payload["backend"] == "dummy"
        assert payload["supports_streaming"] is True


class TestRegistry:
    def test_known_backends(self) -> None:
        assert {"dummy", "indicf5"} <= set(available_backends())

    def test_builds_the_configured_backend(self) -> None:
        settings = Settings(_env_file=None, tts_backend="dummy", sample_rate=16_000)
        synthesizer = build_synthesizer(settings)
        assert synthesizer.info.backend == "dummy"
        assert synthesizer.info.sample_rate == 16_000

    def test_backend_is_not_loaded_on_construction(self) -> None:
        settings = Settings(_env_file=None, tts_backend="dummy")
        assert not build_synthesizer(settings).is_ready()

    def test_unknown_backend_is_refused(self) -> None:
        settings = Settings(_env_file=None, tts_backend="nonexistent")
        with pytest.raises(ConfigurationError, match="unknown tts backend"):
            build_synthesizer(settings)

    def test_duplicate_registration_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="already registered"):
            register_backend("dummy", lambda _: DummySynthesizer())
