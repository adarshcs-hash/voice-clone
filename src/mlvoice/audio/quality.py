"""Ingest quality gates.

Corpus quality dominates TTS output quality, and the failures that matter are
mundane: a clip that is mostly silence, a recording made two metres from the
mic, a file that was normalised into clipping, a DC offset from a bad interface.
Each is cheap to detect and expensive to leave in.

The measurements here are deliberately dependency-free (NumPy only) so they can
run over a million-file corpus without a model, and so that the same gate runs
on user-uploaded reference audio at request time. A perceptual MOS estimator
(DNSMOS, UTMOS) is a better final filter and slots in through
:mod:`mlvoice.eval.metrics`; these gates are the first pass that removes the
obvious damage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from mlvoice.audio.io import Audio
from mlvoice.audio.loudness import peak_dbfs

__all__ = ["QualityGate", "QualityReport", "measure_quality"]

_FRAME_MS: Final = 20.0
_CLIP_THRESHOLD: Final = 0.99
_EPSILON: Final = 1e-12


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Objective measurements of one clip."""

    duration_seconds: float
    peak_dbfs: float
    clipping_ratio: float
    """Fraction of samples at or beyond full scale."""
    dc_offset: float
    """Mean sample value; a non-zero mean indicates a DC offset."""
    silence_ratio: float
    """Fraction of frames below the estimated noise floor plus a margin."""
    estimated_snr_db: float
    """Difference between speech-active and noise-floor frame energies."""

    def as_dict(self) -> dict[str, float]:
        """Flat mapping, for manifests and logs."""
        return {
            "duration_seconds": round(self.duration_seconds, 4),
            "peak_dbfs": round(self.peak_dbfs, 2),
            "clipping_ratio": round(self.clipping_ratio, 6),
            "dc_offset": round(self.dc_offset, 6),
            "silence_ratio": round(self.silence_ratio, 4),
            "estimated_snr_db": round(self.estimated_snr_db, 2),
        }


@dataclass(frozen=True, slots=True)
class QualityGate:
    """Thresholds a clip must satisfy to enter the corpus.

    Defaults are tuned for TTS training data, which is stricter than ASR data:
    a recording good enough to transcribe is often not good enough to imitate.
    """

    min_duration_seconds: float = 0.5
    max_duration_seconds: float = 30.0
    min_snr_db: float = 20.0
    max_clipping_ratio: float = 0.001
    max_silence_ratio: float = 0.6
    max_dc_offset: float = 0.01
    min_peak_dbfs: float = -40.0

    def evaluate(self, report: QualityReport) -> tuple[bool, list[str]]:
        """Return whether ``report`` passes, and the reasons it does not."""
        reasons: list[str] = []
        if report.duration_seconds < self.min_duration_seconds:
            reasons.append(f"too short ({report.duration_seconds:.2f}s)")
        if report.duration_seconds > self.max_duration_seconds:
            reasons.append(f"too long ({report.duration_seconds:.2f}s)")
        if report.estimated_snr_db < self.min_snr_db:
            reasons.append(f"snr too low ({report.estimated_snr_db:.1f} dB)")
        if report.clipping_ratio > self.max_clipping_ratio:
            reasons.append(f"clipped ({report.clipping_ratio:.4%} of samples)")
        if report.silence_ratio > self.max_silence_ratio:
            reasons.append(f"mostly silence ({report.silence_ratio:.0%})")
        if abs(report.dc_offset) > self.max_dc_offset:
            reasons.append(f"dc offset ({report.dc_offset:+.4f})")
        if report.peak_dbfs < self.min_peak_dbfs:
            reasons.append(f"level too low (peak {report.peak_dbfs:.1f} dBFS)")
        return (not reasons), reasons


def _frame_energies(audio: Audio) -> np.ndarray:
    """Per-frame RMS energy in dB, over 20 ms non-overlapping frames."""
    frame = max(1, int(audio.sample_rate * _FRAME_MS / 1000.0))
    usable = (audio.num_samples // frame) * frame
    if usable == 0:
        return np.array([_to_db(float(np.sqrt(np.mean(audio.samples**2) + _EPSILON)))])
    frames = audio.samples[:usable].reshape(-1, frame).astype(np.float64)
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + _EPSILON)
    energies: np.ndarray = 20.0 * np.log10(rms)
    return energies


def _to_db(value: float) -> float:
    return float(20.0 * np.log10(max(value, _EPSILON)))


def _noise_floor_db(energies: np.ndarray) -> float:
    """Estimate the noise floor from a frame-energy sequence.

    Minimum statistics with an outlier guard: the frame energies are divided
    into 0.5 s windows, the minimum of each window is taken, and the noise floor
    is the 10th percentile of those minima.

    Taking window minima rather than a percentile over all frames is what makes
    the estimate correct for continuous speech, where a plain percentile lands
    in a quiet syllable instead of in silence and understates SNR by 20 dB or
    more. Taking a percentile *of the minima* rather than the global minimum is
    what stops a single dropped frame from overstating it.

    Clips too short to hold two windows fall back to the 5th percentile of
    frames.
    """
    window = 25  # 25 x 20 ms = 0.5 s
    if len(energies) >= 2 * window:
        usable = (len(energies) // window) * window
        minima = energies[:usable].reshape(-1, window).min(axis=1)
        return float(np.percentile(minima, 10))
    return float(np.percentile(energies, 5))


def measure_quality(audio: Audio) -> QualityReport:
    """Measure ``audio`` against the objective ingest criteria.

    The SNR estimate needs the clip to contain some non-speech, so measure
    *before* trimming leading and trailing silence; :mod:`mlvoice.data.prepare`
    does exactly that and then substitutes the post-trim duration.
    """
    energies = _frame_energies(audio)
    noise_floor = _noise_floor_db(energies)
    speech_level = float(np.percentile(energies, 90))
    snr = speech_level - noise_floor

    silence_threshold = noise_floor + 6.0
    silence_ratio = float(np.mean(energies < silence_threshold))

    clipping = float(np.mean(np.abs(audio.samples) >= _CLIP_THRESHOLD))
    return QualityReport(
        duration_seconds=audio.duration_seconds,
        peak_dbfs=peak_dbfs(audio),
        clipping_ratio=clipping,
        dc_offset=float(np.mean(audio.samples)),
        silence_ratio=silence_ratio,
        estimated_snr_db=snr,
    )
