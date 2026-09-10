"""Deterministic stub backend.

Not a model. It generates a formant-ish buzz whose duration, pitch contour and
segmentation are derived from the phoneme sequence, which is enough to exercise
everything around the model: the API contract, streaming, chunking, pause
insertion, watermarking, rate limits and the eval harness plumbing.

It exists because the alternative -- requiring multi-gigabyte weights and a GPU
to run the test suite -- means the test suite does not get run. Output is
deterministic given a seed, so API responses are byte-comparable across runs.

:class:`~mlvoice.config.Settings` refuses to select this backend when
``MLVOICE_ENV=production``.
"""

from __future__ import annotations

import hashlib
from typing import Final

import numpy as np

from mlvoice.audio.io import Audio
from mlvoice.tts.base import SynthesisRequest, Synthesizer, SynthesizerInfo

__all__ = ["DummySynthesizer"]

_SECONDS_PER_PHONEME: Final = 0.085
_VOWELS: Final = frozenset("aeiouɨ")
_BASE_F0: Final = 130.0


class DummySynthesizer(Synthesizer):
    """Phoneme-driven test signal generator.

    Args:
        sample_rate: Output rate; match the real backend so downstream code
            paths are identical.
        seed: Base seed. Combined with the chunk text so that identical text
            yields identical audio.
    """

    def __init__(self, *, sample_rate: int = 24_000, seed: int = 0) -> None:
        self._sample_rate = sample_rate
        self._seed = seed
        self._loaded = False

    @property
    def info(self) -> SynthesizerInfo:
        """Static description of this backend."""
        return SynthesizerInfo(
            backend="dummy",
            model_id="dummy",
            revision=None,
            sample_rate=self._sample_rate,
            device="cpu",
            supports_cloning=True,
            supports_streaming=True,
        )

    def load(self) -> None:
        """No weights to load; marks the backend ready."""
        self._loaded = True

    def is_ready(self) -> bool:
        """True once :meth:`load` has been called."""
        return self._loaded

    def _rng(self, chunk_text: str, request: SynthesisRequest) -> np.random.Generator:
        material = f"{self._seed}:{request.seed}:{chunk_text}".encode()
        digest = hashlib.sha256(material).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "big"))

    def _synthesize_chunk(self, chunk_text: str, request: SynthesisRequest) -> Audio:
        """Generate a deterministic buzz shaped by the chunk's phonemes."""
        from mlvoice.text.g2p import phonemize

        phonemes = [p for word in phonemize(chunk_text) for p in word]
        if not phonemes:
            phonemes = ["a"] * max(1, len(chunk_text) // 3)

        rng = self._rng(chunk_text, request)
        # A voice-dependent pitch offset, so two voices are audibly different.
        voice_offset = 0.0
        if request.prompt is not None:
            digest = hashlib.sha256(request.prompt.voice_id.encode()).digest()
            voice_offset = (digest[0] / 255.0 - 0.5) * 80.0

        segments: list[np.ndarray] = []
        for index, phoneme in enumerate(phonemes):
            duration = _SECONDS_PER_PHONEME / request.speed
            if phoneme.endswith("ː"):
                duration *= 1.6
            count = max(1, int(self._sample_rate * duration))
            t = np.arange(count, dtype=np.float32) / self._sample_rate
            wave: np.ndarray
            if phoneme[0] in _VOWELS:
                # Declining pitch contour across the chunk, as speech has.
                f0 = _BASE_F0 + voice_offset - 12.0 * index / max(1, len(phonemes))
                wave = 0.32 * np.sin(2 * np.pi * f0 * t) + 0.10 * np.sin(2 * np.pi * 2 * f0 * t)
            else:
                wave = 0.06 * rng.standard_normal(count).astype(np.float32)
            envelope = np.hanning(count).astype(np.float32) if count > 1 else np.ones(1, np.float32)
            segments.append((wave.astype(np.float32) * envelope).astype(np.float32))

        return Audio(samples=np.concatenate(segments), sample_rate=self._sample_rate)
