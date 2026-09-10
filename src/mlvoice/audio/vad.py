"""Voice activity detection and silence trimming.

Two implementations behind one interface:

*   :class:`EnergyVad` -- an adaptive-threshold energy detector with hangover.
    Dependency-free, deterministic, fast enough for a whole corpus, and good
    enough for trimming leading and trailing silence from studio recordings.
*   :class:`SileroVad` -- the Silero neural VAD. Substantially better on noisy,
    reverberant or far-field audio, and what should run over found data.

Trimming matters more than it sounds. Leading silence teaches an autoregressive
model to begin with silence, which shows up at inference as a long dead pause
before every utterance; trailing silence teaches it not to stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import numpy as np

from mlvoice.audio.io import Audio
from mlvoice.errors import AudioError
from mlvoice.logging import get_logger

__all__ = ["EnergyVad", "SileroVad", "SpeechSegment", "Vad", "trim_silence"]

log = get_logger(__name__)

_EPSILON: Final = 1e-12


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """A detected span of speech, in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        """Span length in seconds."""
        return self.end - self.start


@runtime_checkable
class Vad(Protocol):
    """Anything that can locate speech in an audio buffer."""

    def detect(self, audio: Audio) -> list[SpeechSegment]:
        """Return speech spans in reading order."""
        ...


@dataclass(frozen=True, slots=True)
class EnergyVad:
    """Adaptive energy-threshold VAD.

    The threshold is the estimated noise floor (10th percentile of frame
    energies) plus ``threshold_offset_db``, which adapts to each recording's own
    level rather than assuming a fixed one. ``hangover_ms`` keeps short pauses
    inside a segment so that a stop consonant does not split a word.

    Attributes:
        frame_ms: Analysis frame length.
        threshold_offset_db: How far above the noise floor counts as speech.
        hangover_ms: Trailing tolerance before closing a segment.
        min_speech_ms: Segments shorter than this are discarded as transients.
    """

    frame_ms: float = 20.0
    threshold_offset_db: float = 10.0
    hangover_ms: float = 180.0
    min_speech_ms: float = 100.0

    def detect(self, audio: Audio) -> list[SpeechSegment]:
        """Locate speech spans by frame energy."""
        frame = max(1, int(audio.sample_rate * self.frame_ms / 1000.0))
        usable = (audio.num_samples // frame) * frame
        if usable == 0:
            return []
        frames = audio.samples[:usable].reshape(-1, frame).astype(np.float64)
        energies = 20.0 * np.log10(np.sqrt(np.mean(np.square(frames), axis=1) + _EPSILON))

        noise_floor = float(np.percentile(energies, 10))
        threshold = noise_floor + self.threshold_offset_db
        active = energies > threshold

        hangover_frames = int(self.hangover_ms / self.frame_ms)
        segments: list[SpeechSegment] = []
        start: int | None = None
        idle = 0
        for index, is_active in enumerate(active):
            if is_active:
                if start is None:
                    start = index
                idle = 0
            elif start is not None:
                idle += 1
                if idle > hangover_frames:
                    segments.append(self._to_segment(start, index - idle, frame, audio))
                    start = None
                    idle = 0
        if start is not None:
            segments.append(self._to_segment(start, len(active), frame, audio))

        minimum = self.min_speech_ms / 1000.0
        return [s for s in segments if s.duration >= minimum]

    @staticmethod
    def _to_segment(start_frame: int, end_frame: int, frame: int, audio: Audio) -> SpeechSegment:
        rate = audio.sample_rate
        return SpeechSegment(
            start=start_frame * frame / rate,
            end=min(audio.duration_seconds, max(end_frame, start_frame + 1) * frame / rate),
        )


class SileroVad:
    """Silero neural VAD.

    Loads lazily and once, because model construction dominates per-file cost in
    a corpus run. Requires the ``data`` extra.
    """

    def __init__(self, *, threshold: float = 0.5, min_speech_ms: int = 100) -> None:
        self._threshold = threshold
        self._min_speech_ms = min_speech_ms
        self._model: object | None = None
        self._get_timestamps: object | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise AudioError(
                "SileroVad requires the 'data' extra: pip install 'mlvoice[data]'"
            ) from exc
        self._model = load_silero_vad()
        self._get_timestamps = get_speech_timestamps

    def detect(self, audio: Audio) -> list[SpeechSegment]:
        """Locate speech spans with Silero.

        Raises:
            AudioError: The sample rate is unsupported, or the extra is missing.
        """
        if audio.sample_rate not in {8_000, 16_000}:
            raise AudioError(
                "Silero VAD expects 8 kHz or 16 kHz; resample first",
                sample_rate=audio.sample_rate,
            )
        self._ensure_loaded()
        import torch

        assert self._get_timestamps is not None
        stamps = self._get_timestamps(  # type: ignore[operator]
            torch.from_numpy(audio.samples),
            self._model,
            sampling_rate=audio.sample_rate,
            threshold=self._threshold,
            min_speech_duration_ms=self._min_speech_ms,
            return_seconds=True,
        )
        return [SpeechSegment(start=float(s["start"]), end=float(s["end"])) for s in stamps]


def trim_silence(
    audio: Audio,
    *,
    vad: Vad | None = None,
    pad_ms: float = 50.0,
) -> Audio:
    """Trim leading and trailing silence, keeping ``pad_ms`` on each side.

    Interior pauses are preserved: they are prosody, not noise. Only the outer
    edges are cut.

    Args:
        audio: Input audio.
        vad: Detector to use. Defaults to :class:`EnergyVad`.
        pad_ms: Padding retained around the detected speech extent.

    Returns:
        The trimmed audio, or the input unchanged if no speech was found.
    """
    detector = vad if vad is not None else EnergyVad()
    segments = detector.detect(audio)
    if not segments:
        log.warning("no speech detected; leaving audio untrimmed")
        return audio
    pad = pad_ms / 1000.0
    return audio.slice_seconds(
        max(0.0, segments[0].start - pad), min(audio.duration_seconds, segments[-1].end + pad)
    )
