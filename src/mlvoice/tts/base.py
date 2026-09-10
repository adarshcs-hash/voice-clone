"""Synthesis backend contract.

Every backend -- the reference-prompt IndicF5 model, a future in-house model, a
deterministic stub for tests -- implements :class:`Synthesizer`. The service
layer depends only on this protocol, so swapping the model is a configuration
change rather than a code change, and A/B comparing two models is running two
instances behind the same interface.

Design notes
------------
*   **A reference prompt, not a speaker embedding.** The strong open Malayalam
    models (the F5/E2 flow-matching family) clone from a short reference clip
    *plus its transcript*, and quality depends heavily on that transcript being
    correct. :class:`ReferencePrompt` therefore carries both and refuses to be
    constructed without them.
*   **Chunk-wise generation is the default.** Long-form input is split by the
    text frontend, so :meth:`Synthesizer.stream` yields one buffer per chunk and
    :meth:`Synthesizer.synthesize` is defined in terms of it. That makes
    time-to-first-audio a property of the first chunk rather than of the whole
    request.
*   **Real-time factor is always reported.** RTF is the number that decides
    whether a deployment is affordable, so it is part of the result rather than
    something to reconstruct from logs.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from mlvoice.audio.io import Audio
from mlvoice.errors import ValidationError
from mlvoice.text.pipeline import ProcessedText

__all__ = [
    "ReferencePrompt",
    "SynthesisRequest",
    "SynthesisResult",
    "Synthesizer",
    "SynthesizerInfo",
]

MIN_REFERENCE_SECONDS: Final = 2.0
MAX_REFERENCE_SECONDS: Final = 30.0


@dataclass(frozen=True, slots=True)
class ReferencePrompt:
    """The reference clip a cloning backend conditions on.

    Attributes:
        audio: The reference recording.
        text: Its verbatim transcript, in Malayalam script. A wrong or missing
            transcript is the most common cause of a bad clone with these
            models, so it is required.
        voice_id: Identifier of the enrolled voice, for logging and cache keys.
    """

    audio: Audio
    text: str
    voice_id: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValidationError(
                "a reference prompt requires its transcript", voice_id=self.voice_id
            )
        duration = self.audio.duration_seconds
        if duration < MIN_REFERENCE_SECONDS:
            raise ValidationError(
                "reference audio is too short to characterise a voice",
                seconds=round(duration, 2),
                minimum=MIN_REFERENCE_SECONDS,
            )
        if duration > MAX_REFERENCE_SECONDS:
            raise ValidationError(
                "reference audio is longer than the backend conditions on",
                seconds=round(duration, 2),
                maximum=MAX_REFERENCE_SECONDS,
            )


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    """One synthesis job.

    Attributes:
        text: The processed text frontend output. Passing
            :class:`~mlvoice.text.pipeline.ProcessedText` rather than a raw
            string means the backend cannot accidentally bypass normalisation.
        prompt: Reference prompt for cloning, or ``None`` for a backend's
            default voice.
        speed: Playback rate multiplier applied by the backend, where supported.
        seed: Sampling seed. Fixing it makes a generation reproducible, which
            regression tests depend on.
    """

    text: ProcessedText
    prompt: ReferencePrompt | None = None
    speed: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.5 <= self.speed <= 2.0:
            raise ValidationError("speed must be between 0.5 and 2.0", speed=self.speed)


@dataclass(frozen=True, slots=True)
class SynthesizerInfo:
    """Static description of a loaded backend, surfaced by the health endpoint."""

    backend: str
    model_id: str
    revision: str | None
    sample_rate: int
    device: str
    supports_cloning: bool
    supports_streaming: bool

    def as_dict(self) -> dict[str, object]:
        """Flat mapping for the health payload."""
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "revision": self.revision,
            "sample_rate": self.sample_rate,
            "device": self.device,
            "supports_cloning": self.supports_cloning,
            "supports_streaming": self.supports_streaming,
        }


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """Generated audio and how it was produced."""

    audio: Audio
    info: SynthesizerInfo
    chunk_count: int
    generation_seconds: float
    voice_id: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def real_time_factor(self) -> float:
        """Generation time divided by audio duration. Below 1.0 is faster than real time."""
        duration = self.audio.duration_seconds
        return self.generation_seconds / duration if duration > 0 else float("inf")

    def metrics(self) -> dict[str, object]:
        """Log-safe performance summary."""
        return {
            "audio_seconds": round(self.audio.duration_seconds, 3),
            "generation_seconds": round(self.generation_seconds, 3),
            "real_time_factor": round(self.real_time_factor, 3),
            "chunks": self.chunk_count,
            "backend": self.info.backend,
        }


class Synthesizer(ABC):
    """Base class for synthesis backends.

    Subclasses implement :meth:`_synthesize_chunk` and :meth:`info`; chunk
    iteration, silence insertion, concatenation and timing are handled here so
    that every backend behaves identically at the seams.
    """

    @property
    @abstractmethod
    def info(self) -> SynthesizerInfo:
        """Static description of this backend."""

    @abstractmethod
    def load(self) -> None:
        """Load weights. Idempotent, and called before serving traffic."""

    @abstractmethod
    def is_ready(self) -> bool:
        """True once :meth:`load` has completed successfully."""

    @abstractmethod
    def _synthesize_chunk(self, chunk_text: str, request: SynthesisRequest) -> Audio:
        """Generate audio for one already-chunked piece of text."""

    def stream(self, request: SynthesisRequest) -> Iterator[Audio]:
        """Yield one audio buffer per text chunk, followed by its pause.

        The pause is emitted as part of the same buffer so that a client writing
        buffers straight to an output device gets correct timing without having
        to interpret metadata.
        """
        for chunk in request.text.chunks:
            audio = self._synthesize_chunk(chunk.text, request)
            if chunk.pause_ms > 0:
                silence = np.zeros(
                    int(audio.sample_rate * chunk.pause_ms / 1000.0), dtype=np.float32
                )
                audio = audio.with_samples(np.concatenate([audio.samples, silence]))
            yield audio

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Generate the whole utterance.

        Raises:
            ValidationError: The request produced no chunks, which means the
                text frontend rejected the input.
        """
        started = time.perf_counter()
        buffers = list(self.stream(request))
        elapsed = time.perf_counter() - started
        if not buffers:
            raise ValidationError("nothing to synthesise")
        samples = np.concatenate([b.samples for b in buffers])
        return SynthesisResult(
            audio=Audio(samples=samples, sample_rate=buffers[0].sample_rate),
            info=self.info,
            chunk_count=len(buffers),
            generation_seconds=elapsed,
            voice_id=request.prompt.voice_id if request.prompt else None,
        )
