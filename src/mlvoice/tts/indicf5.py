"""IndicF5 backend.

`ai4bharat/IndicF5 <https://huggingface.co/ai4bharat/IndicF5>`_ is an F5-TTS
(flow-matching) model trained on 1,417 hours across 11 Indian languages,
Malayalam included. It clones zero-shot from a reference clip plus that clip's
transcript, which makes it the strongest open starting point for a Malayalam
voice product and the model this project targets first.

Operational notes
-----------------
*   **The reference transcript matters.** The model conditions on ``ref_text``;
    supplying the wrong transcript degrades the clone badly. Enrolment stores a
    verified transcript for exactly this reason.
*   **``trust_remote_code`` is required**, because the model ships custom
    modelling code. That is remote code execution by design, so the revision is
    pinned: :class:`~mlvoice.config.Settings` refuses an unpinned revision in
    production, and this backend passes the pin through to the hub.
*   **The model takes a file path**, not an array, so the reference clip is
    written to a private temporary file per request and removed afterwards.
*   Output is 24 kHz and may arrive as ``int16`` or ``float32``; both are
    normalised to ``float32`` in ``[-1, 1]`` here.

This module is excluded from coverage: it cannot run without multi-gigabyte
weights. Its contract is tested through :class:`~mlvoice.tts.base.Synthesizer`
against the stub backend, and the real path is exercised by the ``slow``-marked
tests.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Final

import numpy as np

from mlvoice.audio.io import Audio, resample, save_audio
from mlvoice.errors import BackendUnavailableError, ValidationError
from mlvoice.logging import get_logger
from mlvoice.tts.base import SynthesisRequest, Synthesizer, SynthesizerInfo

__all__ = ["IndicF5Synthesizer"]

log = get_logger(__name__)

MODEL_SAMPLE_RATE: Final = 24_000
_INT16_SCALE: Final = 32768.0


class IndicF5Synthesizer(Synthesizer):
    """Reference-prompt Malayalam synthesis with IndicF5.

    Args:
        model_id: Hub repository id.
        revision: Commit or tag to pin. Strongly recommended -- the backend
            executes remote code from this repository.
        device: ``cpu``, ``cuda`` or ``mps``.
        default_prompt: Reference clip used when a request carries no prompt.
    """

    def __init__(
        self,
        *,
        model_id: str = "ai4bharat/IndicF5",
        revision: str | None = None,
        device: str = "cpu",
        default_prompt: tuple[Path, str] | None = None,
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._device = device
        self._default_prompt = default_prompt
        self._model: Any | None = None

    @property
    def info(self) -> SynthesizerInfo:
        """Static description of this backend."""
        return SynthesizerInfo(
            backend="indicf5",
            model_id=self._model_id,
            revision=self._revision,
            sample_rate=MODEL_SAMPLE_RATE,
            device=self._device,
            supports_cloning=True,
            supports_streaming=True,
        )

    def load(self) -> None:
        """Load the model from the hub. Idempotent.

        Raises:
            BackendUnavailableError: The ``models`` extra is missing, or the
                weights cannot be loaded.
        """
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise BackendUnavailableError(
                "IndicF5 requires the 'models' extra: pip install 'mlvoice[models]'"
            ) from exc

        if self._revision is None:
            log.warning(
                "loading a model with trust_remote_code from an unpinned revision",
                model_id=self._model_id,
            )
        try:
            model = AutoModel.from_pretrained(
                self._model_id,
                revision=self._revision,
                trust_remote_code=True,
            )
            self._model = model.to(torch.device(self._device)).eval()
        except Exception as exc:
            raise BackendUnavailableError(
                "could not load the IndicF5 weights",
                model_id=self._model_id,
                revision=self._revision,
                reason=str(exc),
            ) from exc
        log.info(
            "indicf5 loaded", model_id=self._model_id, revision=self._revision, device=self._device
        )

    def is_ready(self) -> bool:
        """True once the weights are resident."""
        return self._model is not None

    def _synthesize_chunk(self, chunk_text: str, request: SynthesisRequest) -> Audio:
        """Generate one chunk, conditioning on the request's reference prompt.

        Raises:
            BackendUnavailableError: :meth:`load` has not run.
            ValidationError: No prompt was supplied and no default is configured.
        """
        if self._model is None:
            raise BackendUnavailableError("IndicF5 backend is not loaded")

        if request.prompt is not None:
            with tempfile.TemporaryDirectory(prefix="mlvoice-ref-") as directory:
                reference_path = Path(directory) / "reference.wav"
                save_audio(resample(request.prompt.audio, MODEL_SAMPLE_RATE), reference_path)
                raw = self._invoke(chunk_text, reference_path, request.prompt.text)
        elif self._default_prompt is not None:
            raw = self._invoke(chunk_text, self._default_prompt[0], self._default_prompt[1])
        else:
            raise ValidationError(
                "IndicF5 needs a reference prompt; enrol a voice or configure a default"
            )
        return Audio(samples=_to_float32(raw), sample_rate=MODEL_SAMPLE_RATE)

    def _invoke(self, text: str, reference_path: Path, reference_text: str) -> np.ndarray:
        import torch

        assert self._model is not None
        with torch.inference_mode():
            output = self._model(
                text,
                ref_audio_path=str(reference_path),
                ref_text=reference_text,
            )
        return np.asarray(output)


def _to_float32(raw: np.ndarray) -> np.ndarray:
    """Normalise model output to ``float32`` in ``[-1, 1]``."""
    array = np.asarray(raw)
    if array.dtype == np.int16:
        array = array.astype(np.float32) / _INT16_SCALE
    array = array.astype(np.float32, copy=False).reshape(-1)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 1.0:
        array = array / peak
    return array
