"""Audio watermarking.

Every clip this service generates is watermarked. That is a product requirement,
not a feature: a voice-cloning system without an attribution path is a
deniability machine, and the ability to answer "did we generate this?" is what
makes an abuse report actionable.

Two implementations:

*   :class:`AudioSealWatermarker` -- Meta's AudioSeal. A trained embedder and
    detector, robust to compression, resampling, filtering and cropping. This is
    what production should run; it needs the ``models`` extra.
*   :class:`SpreadSpectrumWatermarker` -- a direct-sequence spread-spectrum
    watermark implemented here in NumPy. Its honest limits: it survives lossless
    handling, gain changes and format conversion, and it does **not** survive
    lossy compression, aggressive resampling, or time-stretching. It exists so
    that a default install still watermarks, and so the plumbing is testable.

The payload is a 32-bit identifier, not the text. It is a key into the request
log, so no content leaves in the audio.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

from mlvoice.audio.io import Audio
from mlvoice.errors import ValidationError
from mlvoice.logging import get_logger

__all__ = [
    "AudioSealWatermarker",
    "SpreadSpectrumWatermarker",
    "WatermarkDetection",
    "Watermarker",
    "payload_for",
]

log = get_logger(__name__)

PAYLOAD_BITS: Final = 32
_DEFAULT_STRENGTH: Final = 0.008
"""Embedding amplitude relative to full scale, about -42 dBFS."""
_BAND_LO_RATIO: Final = 0.58
_BAND_HI_RATIO: Final = 0.92
"""Watermark band, as a fraction of Nyquist. For 24 kHz audio this is roughly
7-11 kHz: above almost all speech energy, below the anti-alias roll-off."""


@dataclass(frozen=True, slots=True)
class WatermarkDetection:
    """Outcome of a detection attempt."""

    detected: bool
    payload: int | None
    confidence: float
    """Mean normalised correlation magnitude across payload bits, in ``[0, 1]``."""

    def as_dict(self) -> dict[str, object]:
        """Flat mapping for API responses."""
        return {
            "detected": self.detected,
            "payload": self.payload,
            "confidence": round(self.confidence, 4),
        }


@runtime_checkable
class Watermarker(Protocol):
    """Anything that can embed and detect an identifier in speech."""

    @property
    def name(self) -> str:
        """Identifier recorded alongside generated audio."""
        ...

    def embed(self, audio: Audio, payload: int) -> Audio:
        """Return ``audio`` carrying ``payload``."""
        ...

    def detect(self, audio: Audio) -> WatermarkDetection:
        """Attempt to recover a payload from ``audio``."""
        ...


def payload_for(*parts: str) -> int:
    """Derive a stable 32-bit payload from identifying strings.

    Use the request id and voice id, so a reported clip maps back to one log
    entry without the audio carrying anything about its content.
    """
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:4], "big")


class SpreadSpectrumWatermarker:
    """Direct-sequence spread-spectrum watermark.

    The signal is divided into :data:`PAYLOAD_BITS` equal blocks. Each block
    receives ``±strength * chip``, where ``chip`` is a keyed pseudo-random ±1
    sequence modulated by a high-frequency carrier; the sign carries the bit.
    Detection correlates each block against its own chip and reads the sign.

    Correlation is normalised by block energy, so the watermark survives an
    arbitrary gain change -- including the loudness normalisation this service
    applies after embedding.

    The chips are band-limited to :data:`_BAND_LO_RATIO`-:data:`_BAND_HI_RATIO`
    of Nyquist, and detection band-passes the signal to the same range before
    correlating. That is what makes the scheme work at an inaudible amplitude:
    speech carries little energy up there, so the interference the detector has
    to overcome is small.

    Args:
        key: Secret used to derive the chip sequences. Detection needs the same
            key, so treat it as a long-lived deployment secret.
        strength: Embedding amplitude. Raising it improves robustness and
            eventually becomes audible.
        detection_threshold: Minimum mean normalised in-band correlation to
            report a detection.
    """

    def __init__(
        self,
        key: bytes,
        *,
        strength: float = _DEFAULT_STRENGTH,
        detection_threshold: float = 0.06,
    ) -> None:
        if not key:
            raise ValidationError("watermark key must not be empty")
        self._key = key
        self._strength = strength
        self._threshold = detection_threshold

    @property
    def name(self) -> str:
        """Identifier recorded alongside generated audio."""
        return "spread-spectrum"

    @staticmethod
    def _band_bins(length: int) -> tuple[int, int]:
        """Return the rFFT bin range covering the watermark band."""
        bins = length // 2 + 1
        return int(_BAND_LO_RATIO * (bins - 1)), int(_BAND_HI_RATIO * (bins - 1))

    def _chip(self, bit_index: int, length: int) -> np.ndarray:
        """Keyed, band-limited, unit-RMS chip sequence for one payload bit.

        Built in the frequency domain -- random phases inside the watermark band
        and zeros outside -- so the chip is confined to the band the detector
        looks at.
        """
        seed = hmac.new(self._key, f"chip:{bit_index}".encode(), hashlib.sha256).digest()
        rng = np.random.default_rng(int.from_bytes(seed[:8], "big"))
        spectrum = np.zeros(length // 2 + 1, dtype=np.complex128)
        lo, hi = self._band_bins(length)
        phases = rng.uniform(0.0, 2 * np.pi, size=hi - lo)
        spectrum[lo:hi] = np.exp(1j * phases)
        chip = np.fft.irfft(spectrum, n=length)
        rms = float(np.sqrt(np.mean(np.square(chip))))
        if rms == 0.0:  # pragma: no cover - only for degenerate block sizes
            return np.zeros(length, dtype=np.float32)
        return (chip / rms).astype(np.float32)

    def _bandpass(self, block: np.ndarray) -> np.ndarray:
        """Zero every rFFT bin outside the watermark band."""
        length = len(block)
        spectrum = np.fft.rfft(block)
        lo, hi = self._band_bins(length)
        mask = np.zeros_like(spectrum)
        mask[lo:hi] = spectrum[lo:hi]
        return np.fft.irfft(mask, n=length)

    @staticmethod
    def _blocks(length: int) -> list[tuple[int, int]]:
        """Split ``length`` samples into equal payload blocks."""
        size = length // PAYLOAD_BITS
        return [(i * size, (i + 1) * size) for i in range(PAYLOAD_BITS)]

    def embed(self, audio: Audio, payload: int) -> Audio:
        """Embed ``payload`` into ``audio``.

        Raises:
            ValidationError: The clip is too short to hold the payload, or the
                payload does not fit in :data:`PAYLOAD_BITS`.
        """
        if not 0 <= payload < 2**PAYLOAD_BITS:
            raise ValidationError("payload does not fit in 32 bits", payload=payload)
        block_size = audio.num_samples // PAYLOAD_BITS
        if block_size < 256:
            raise ValidationError(
                "audio is too short to watermark",
                samples=audio.num_samples,
                minimum=256 * PAYLOAD_BITS,
            )

        out = audio.samples.copy()
        for index, (start, end) in enumerate(self._blocks(audio.num_samples)):
            bit = (payload >> (PAYLOAD_BITS - 1 - index)) & 1
            sign = 1.0 if bit else -1.0
            chip = self._chip(index, end - start)
            out[start:end] += (sign * self._strength * chip).astype(np.float32)
        return audio.with_samples(np.clip(out, -1.0, 1.0))

    def detect(self, audio: Audio) -> WatermarkDetection:
        """Attempt to recover a payload from ``audio``."""
        block_size = audio.num_samples // PAYLOAD_BITS
        if block_size < 256:
            return WatermarkDetection(detected=False, payload=None, confidence=0.0)

        payload = 0
        magnitudes: list[float] = []
        for index, (start, end) in enumerate(self._blocks(audio.num_samples)):
            block = self._bandpass(audio.samples[start:end].astype(np.float64))
            chip = self._chip(index, end - start).astype(np.float64)
            denominator = float(np.linalg.norm(block) * np.linalg.norm(chip))
            correlation = float(np.dot(block, chip) / denominator) if denominator else 0.0
            payload = (payload << 1) | (1 if correlation > 0 else 0)
            magnitudes.append(abs(correlation))

        confidence = float(np.mean(magnitudes))
        detected = confidence >= self._threshold
        return WatermarkDetection(
            detected=detected, payload=payload if detected else None, confidence=confidence
        )


class AudioSealWatermarker:
    """AudioSeal neural watermark.

    Robust to the transformations that matter in the wild -- MP3/AAC encoding,
    resampling, filtering, cropping -- which is why production should use it.
    Requires the ``models`` extra. Loads lazily.
    """

    _MODEL_RATE = 16_000

    def __init__(self, *, generator: str = "audioseal_wm_16bits") -> None:
        self._generator_name = generator
        self._generator: Any | None = None
        self._detector: Any | None = None

    @property
    def name(self) -> str:
        """Identifier recorded alongside generated audio."""
        return "audioseal"

    def _load(self) -> tuple[Any, Any]:
        """Load the generator and detector once, returning both non-optionally."""
        if self._generator is None or self._detector is None:
            try:
                from audioseal import AudioSeal
            except ImportError as exc:  # pragma: no cover - needs the extra
                raise ValidationError(
                    "AudioSealWatermarker requires the 'models' extra plus the audioseal package"
                ) from exc
            self._generator = AudioSeal.load_generator(self._generator_name)
            self._detector = AudioSeal.load_detector("audioseal_detector_16bits")
        return self._generator, self._detector

    def embed(self, audio: Audio, payload: int) -> Audio:  # pragma: no cover - needs the extra
        """Embed the low 16 bits of ``payload`` using AudioSeal."""
        generator, _ = self._load()
        import torch

        from mlvoice.audio.io import resample

        working = resample(audio, self._MODEL_RATE)
        tensor = torch.from_numpy(working.samples).unsqueeze(0).unsqueeze(0)
        bits = torch.tensor([[(payload >> i) & 1 for i in range(16)]], dtype=torch.int32)
        watermark = generator.get_watermark(tensor, self._MODEL_RATE, message=bits)
        out = (tensor + watermark).squeeze(0).squeeze(0).cpu().numpy()
        marked = Audio(samples=out.astype(np.float32), sample_rate=self._MODEL_RATE)
        return resample(marked, audio.sample_rate)

    def detect(self, audio: Audio) -> WatermarkDetection:  # pragma: no cover - needs the extra
        """Detect an AudioSeal watermark."""
        _, detector = self._load()
        import torch

        from mlvoice.audio.io import resample

        working = resample(audio, self._MODEL_RATE)
        tensor = torch.from_numpy(working.samples).unsqueeze(0).unsqueeze(0)
        result, message = detector.detect_watermark(tensor, self._MODEL_RATE)
        confidence = float(result)
        payload = int("".join(str(int(b)) for b in message[0].tolist()[::-1]), 2)
        return WatermarkDetection(
            detected=confidence > 0.5,
            payload=payload if confidence > 0.5 else None,
            confidence=confidence,
        )
