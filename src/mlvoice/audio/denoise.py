"""Speech denoising.

Found data -- broadcast archives, YouTube, field recordings -- carries stationary
noise that a TTS model will faithfully learn to reproduce. Denoising before
training removes it from the target.

Only a real denoiser is offered. There is no spectral-subtraction fallback here
on purpose: naive subtraction leaves musical artefacts that are *worse* for a
generative model than the original noise, because the artefacts are structured
and the model learns them as speech. If the extra is not installed, the
pipeline records the clip as un-denoised and the operator decides, rather than
having quality quietly degraded.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mlvoice.audio.io import Audio, resample
from mlvoice.errors import AudioError
from mlvoice.logging import get_logger

__all__ = ["DeepFilterNetDenoiser", "Denoiser", "NoopDenoiser"]

log = get_logger(__name__)


@runtime_checkable
class Denoiser(Protocol):
    """Anything that can attenuate noise in a speech buffer."""

    def process(self, audio: Audio) -> Audio:
        """Return a denoised copy of ``audio``."""
        ...

    @property
    def name(self) -> str:
        """Identifier recorded in the corpus manifest."""
        ...


class NoopDenoiser:
    """Pass-through denoiser, so the pipeline has no conditional branches."""

    @property
    def name(self) -> str:
        """Identifier recorded in the corpus manifest."""
        return "noop"

    def process(self, audio: Audio) -> Audio:
        """Return ``audio`` unchanged."""
        return audio


class DeepFilterNetDenoiser:
    """DeepFilterNet speech enhancement.

    Operates at 48 kHz internally; input is resampled in and out, which is
    lossless enough at ``soxr`` VHQ quality to be invisible next to the
    enhancement itself. Requires the ``data`` extra.
    """

    _MODEL_RATE = 48_000

    def __init__(self) -> None:
        self._model: object | None = None
        self._state: object | None = None
        self._enhance: object | None = None

    @property
    def name(self) -> str:
        """Identifier recorded in the corpus manifest."""
        return "deepfilternet"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from df.enhance import enhance, init_df
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise AudioError(
                "DeepFilterNetDenoiser requires the 'data' extra: pip install 'mlvoice[data]'"
            ) from exc
        self._model, self._state, _ = init_df()
        self._enhance = enhance

    def process(self, audio: Audio) -> Audio:
        """Denoise ``audio``, preserving its original sample rate.

        Raises:
            AudioError: The extra is not installed.
        """
        self._ensure_loaded()
        import torch

        assert self._enhance is not None
        working = resample(audio, self._MODEL_RATE)
        tensor = torch.from_numpy(working.samples).unsqueeze(0)
        enhanced = self._enhance(self._model, self._state, tensor)  # type: ignore[operator]
        out = Audio(
            samples=enhanced.squeeze(0).cpu().numpy().astype("float32"),
            sample_rate=self._MODEL_RATE,
        )
        return resample(out, audio.sample_rate)
