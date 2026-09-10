"""Loudness measurement and normalisation.

A TTS corpus assembled from several sources arrives at wildly different levels,
and level differences between speakers teach the model to associate a voice with
a gain rather than with a timbre. Every clip is therefore normalised to a single
integrated loudness target before training, and synthesis output is normalised
to the same target so that generated speech matches the corpus it came from.

``-23 LUFS`` is the EBU R 128 broadcast target and the default here; ``-16 LUFS``
is the common streaming target and is what most speech APIs return.

Integrated loudness is measured per ITU-R BS.1770-4 when ``pyloudnorm`` is
installed. Without it, the module falls back to RMS in dBFS, which is *not* the
same quantity -- it ignores K-weighting and gating -- so the fallback is
reported in the result rather than passed off as LUFS.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np

from mlvoice.audio.io import Audio
from mlvoice.errors import AudioError

__all__ = [
    "BROADCAST_TARGET_LUFS",
    "STREAMING_TARGET_LUFS",
    "LoudnessMeasurement",
    "LoudnessMethod",
    "measure_loudness",
    "normalize_loudness",
    "peak_dbfs",
]

BROADCAST_TARGET_LUFS: Final = -23.0
STREAMING_TARGET_LUFS: Final = -16.0
_MIN_LUFS_DURATION: Final = 0.4
"""BS.1770 gating needs at least one 400 ms block."""
_EPSILON: Final = 1e-12


class LoudnessMethod(StrEnum):
    """Which measurement actually produced a value."""

    ITU_BS1770 = "itu_bs1770"
    RMS_DBFS = "rms_dbfs"


@dataclass(frozen=True, slots=True)
class LoudnessMeasurement:
    """A loudness reading and how it was obtained."""

    value: float
    method: LoudnessMethod
    peak_dbfs: float

    @property
    def is_true_lufs(self) -> bool:
        """True when ``value`` is genuine integrated loudness, not an RMS proxy."""
        return self.method is LoudnessMethod.ITU_BS1770


def peak_dbfs(audio: Audio) -> float:
    """Sample peak in dBFS. Returns ``-inf`` for digital silence."""
    peak = float(np.max(np.abs(audio.samples))) if audio.num_samples else 0.0
    if peak <= 0.0:
        return float("-inf")
    return float(20.0 * np.log10(peak))


def _rms_dbfs(audio: Audio) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio.samples, dtype=np.float64)) + _EPSILON))
    return float(20.0 * np.log10(max(rms, _EPSILON)))


def measure_loudness(audio: Audio) -> LoudnessMeasurement:
    """Measure integrated loudness, preferring ITU-R BS.1770-4.

    Clips shorter than 400 ms cannot be gated per the standard and fall back to
    RMS regardless of whether ``pyloudnorm`` is present.
    """
    peak = peak_dbfs(audio)
    if audio.duration_seconds >= _MIN_LUFS_DURATION:
        try:
            import pyloudnorm

            meter = pyloudnorm.Meter(audio.sample_rate)
            value = float(meter.integrated_loudness(audio.samples.astype(np.float64)))
            if np.isfinite(value):
                return LoudnessMeasurement(value, LoudnessMethod.ITU_BS1770, peak)
        except ImportError:
            pass
    return LoudnessMeasurement(_rms_dbfs(audio), LoudnessMethod.RMS_DBFS, peak)


def normalize_loudness(
    audio: Audio,
    *,
    target_lufs: float = BROADCAST_TARGET_LUFS,
    max_peak_dbfs: float = -1.0,
    max_gain_db: float = 30.0,
) -> tuple[Audio, LoudnessMeasurement]:
    """Normalise ``audio`` to ``target_lufs`` without exceeding ``max_peak_dbfs``.

    Gain is applied, then reduced if the result would exceed the peak ceiling,
    so the output never clips. Gain is capped at ``max_gain_db`` so that a clip
    which is effectively silence is not amplified into noise.

    Args:
        audio: Input audio.
        target_lufs: Desired integrated loudness.
        max_peak_dbfs: Sample-peak ceiling after normalisation.
        max_gain_db: Largest gain that may be applied.

    Returns:
        The normalised audio and the measurement taken *before* normalisation.

    Raises:
        AudioError: ``audio`` is digital silence, which has no loudness to
            normalise.
    """
    measurement = measure_loudness(audio)
    if not np.isfinite(measurement.value) or measurement.peak_dbfs == float("-inf"):
        raise AudioError("cannot normalise digital silence")

    gain_db = min(target_lufs - measurement.value, max_gain_db)
    projected_peak = measurement.peak_dbfs + gain_db
    if projected_peak > max_peak_dbfs:
        gain_db -= projected_peak - max_peak_dbfs

    gain = float(10.0 ** (gain_db / 20.0))
    return audio.with_samples(np.clip(audio.samples * gain, -1.0, 1.0)), measurement
