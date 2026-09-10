"""Audio loading, resampling and writing.

One representation is used everywhere inside the process: mono ``float32`` in
``[-1, 1]`` at a known sample rate, carried by :class:`Audio`. Conversions
happen at the edges -- on ingest and on response encoding -- so no downstream
code has to ask whether a buffer is interleaved, integer-scaled or stereo.

Resampling quality is not negotiable for a voice product: linear interpolation
aliases audibly and that aliasing ends up in the training data. The resampler
therefore requires a real implementation (``soxr``, then ``librosa``, then
``torchaudio``) and refuses to silently degrade; the linear fallback exists only
behind an explicit opt-in for tests and quick looks.
"""

from __future__ import annotations

import io as _io
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import soundfile as sf

from mlvoice.errors import AudioError
from mlvoice.logging import get_logger

__all__ = ["Audio", "encode_wav", "load_audio", "resample", "save_audio"]

log = get_logger(__name__)

_MAX_INT16: Final = 32767


@dataclass(frozen=True, slots=True)
class Audio:
    """Mono ``float32`` audio with its sample rate.

    Attributes:
        samples: 1-D ``float32`` array, nominally in ``[-1, 1]``.
        sample_rate: Samples per second.
    """

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise AudioError("Audio expects a 1-D mono array", shape=str(self.samples.shape))
        if self.sample_rate <= 0:
            raise AudioError("sample rate must be positive", sample_rate=self.sample_rate)

    @property
    def duration_seconds(self) -> float:
        """Length in seconds."""
        return len(self.samples) / self.sample_rate

    @property
    def num_samples(self) -> int:
        """Length in samples."""
        return len(self.samples)

    def with_samples(self, samples: np.ndarray) -> Audio:
        """Return a copy carrying ``samples`` at the same rate."""
        return Audio(samples=np.asarray(samples, dtype=np.float32), sample_rate=self.sample_rate)

    def slice_seconds(self, start: float, end: float) -> Audio:
        """Return the ``[start, end)`` second span, clamped to the buffer."""
        a = max(0, int(start * self.sample_rate))
        b = min(self.num_samples, int(end * self.sample_rate))
        if b <= a:
            return self.with_samples(np.zeros(0, dtype=np.float32))
        return self.with_samples(self.samples[a:b])


def _to_mono(data: np.ndarray) -> np.ndarray:
    """Average multi-channel audio down to mono."""
    if data.ndim == 1:
        return data
    mono: np.ndarray = data.mean(axis=1)
    return mono


def load_audio(
    source: str | Path | bytes,
    *,
    target_sample_rate: int | None = None,
    allow_linear_fallback: bool = False,
) -> Audio:
    """Load audio from a path or an in-memory buffer.

    Args:
        source: File path, or the encoded bytes of an audio file.
        target_sample_rate: Resample to this rate when it differs from the file's.
        allow_linear_fallback: Permit linear-interpolation resampling if no
            high-quality resampler is installed. Off by default.

    Returns:
        Mono ``float32`` audio.

    Raises:
        AudioError: The input cannot be decoded, or decodes to nothing.
    """
    try:
        if isinstance(source, bytes):
            data, rate = sf.read(_io.BytesIO(source), dtype="float32", always_2d=False)
        else:
            data, rate = sf.read(str(source), dtype="float32", always_2d=False)
    except Exception as exc:  # soundfile raises a family of errors
        raise AudioError("could not decode audio", reason=str(exc)) from exc

    samples = np.ascontiguousarray(_to_mono(np.asarray(data, dtype=np.float32)))
    if samples.size == 0:
        raise AudioError("audio contains no samples")

    audio = Audio(samples=samples, sample_rate=int(rate))
    if target_sample_rate is not None and target_sample_rate != audio.sample_rate:
        audio = resample(audio, target_sample_rate, allow_linear_fallback=allow_linear_fallback)
    return audio


def resample(
    audio: Audio, target_sample_rate: int, *, allow_linear_fallback: bool = False
) -> Audio:
    """Resample to ``target_sample_rate``.

    Tries ``soxr``, then ``librosa``, then ``torchaudio``. Raises rather than
    degrading quality silently, unless ``allow_linear_fallback`` is set.

    Raises:
        AudioError: No high-quality resampler is available and the fallback was
            not enabled.
    """
    if target_sample_rate == audio.sample_rate:
        return audio
    if target_sample_rate <= 0:
        raise AudioError("target sample rate must be positive", target=target_sample_rate)

    try:
        import soxr

        out = soxr.resample(audio.samples, audio.sample_rate, target_sample_rate, quality="VHQ")
        return Audio(samples=np.asarray(out, dtype=np.float32), sample_rate=target_sample_rate)
    except ImportError:
        pass

    try:
        import librosa

        out = librosa.resample(
            audio.samples,
            orig_sr=audio.sample_rate,
            target_sr=target_sample_rate,
            res_type="soxr_vhq",
        )
        return Audio(samples=np.asarray(out, dtype=np.float32), sample_rate=target_sample_rate)
    except ImportError:
        pass

    try:
        import torch
        import torchaudio.functional as taf

        tensor = torch.from_numpy(audio.samples).unsqueeze(0)
        out_t = taf.resample(tensor, audio.sample_rate, target_sample_rate)
        return Audio(
            samples=out_t.squeeze(0).numpy().astype(np.float32),
            sample_rate=target_sample_rate,
        )
    except ImportError:
        pass

    if not allow_linear_fallback:
        raise AudioError(
            "no high-quality resampler installed; install the 'data' extra "
            "(soxr) or pass allow_linear_fallback=True to accept aliasing",
            source_rate=audio.sample_rate,
            target_rate=target_sample_rate,
        )

    log.warning(
        "resampling with linear interpolation; expect aliasing",
        source_rate=audio.sample_rate,
        target_rate=target_sample_rate,
    )
    ratio = target_sample_rate / audio.sample_rate
    out_len = max(1, round(audio.num_samples * ratio))
    source_positions = np.arange(audio.num_samples, dtype=np.float64)
    target_positions = np.linspace(0, audio.num_samples - 1, out_len)
    out = np.interp(target_positions, source_positions, audio.samples)
    return Audio(samples=out.astype(np.float32), sample_rate=target_sample_rate)


def save_audio(
    audio: Audio,
    path: str | Path,
    *,
    subtype: Literal["PCM_16", "PCM_24", "FLOAT"] = "PCM_16",
) -> Path:
    """Write ``audio`` to ``path``, creating parent directories."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), audio.samples, audio.sample_rate, subtype=subtype)
    return destination


def encode_wav(audio: Audio) -> bytes:
    """Encode ``audio`` as a 16-bit PCM WAV byte string.

    Uses the standard library so that response encoding has no dependency on
    libsndfile's write path, which matters for the streaming endpoint.
    """
    clipped = np.clip(audio.samples, -1.0, 1.0)
    pcm = (clipped * _MAX_INT16).astype("<i2")
    buffer = _io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(audio.sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def wav_header(sample_rate: int, *, data_bytes: int = 0xFFFFFFF) -> bytes:
    """Build a WAV header for streaming, with an effectively unbounded length.

    Chunked responses cannot know the total length up front. Writing a maximal
    size is the conventional trick and is accepted by browsers and ffmpeg.
    """
    header = _io.BytesIO()
    block_align = 2
    header.write(b"RIFF")
    header.write((36 + data_bytes).to_bytes(4, "little"))
    header.write(b"WAVEfmt ")
    header.write((16).to_bytes(4, "little"))
    header.write((1).to_bytes(2, "little"))  # PCM
    header.write((1).to_bytes(2, "little"))  # mono
    header.write(sample_rate.to_bytes(4, "little"))
    header.write((sample_rate * block_align).to_bytes(4, "little"))
    header.write(block_align.to_bytes(2, "little"))
    header.write((16).to_bytes(2, "little"))
    header.write(b"data")
    header.write(data_bytes.to_bytes(4, "little"))
    return header.getvalue()


def to_pcm16(audio: Audio) -> bytes:
    """Encode raw 16-bit little-endian PCM frames without a header."""
    clipped = np.clip(audio.samples, -1.0, 1.0)
    frames: bytes = (clipped * _MAX_INT16).astype("<i2").tobytes()
    return frames
