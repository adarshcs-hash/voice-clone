"""IndicF5 backend.

`ai4bharat/IndicF5 <https://huggingface.co/ai4bharat/IndicF5>`_ is an F5-TTS
(flow-matching) model trained on 1,417 hours across 11 Indian languages,
Malayalam included. It clones zero-shot from a reference clip plus that clip's
transcript, which makes it the strongest open starting point for a Malayalam
voice product and the model this project targets first.

The ``f5_tts`` dependency is not the one on PyPI
------------------------------------------------
IndicF5's remote code imports ``f5_tts``, and there are two different packages
with that import name. The one on PyPI is SWivid's upstream F5-TTS; AI4Bharat
ships its own, at ``github.com/AI4Bharat/IndicF5``, whose API the model's code
actually targets -- upstream's ``load_model`` takes a required ``ckpt_path``
positional argument and AI4Bharat's does not. Install the fork:

.. code-block:: shell

    pip install "mlvoice[indicf5]"

The PyPI package gets far enough to look like it worked and then fails with
``load_model() missing 1 required positional argument``.

Access
------
The repository is **gated**: fetching it requires requesting access on the
model page and then authenticating, or ``load`` fails with a 401 and a
``GatedRepoError``. Grant access to the account whose token the process will
use, and provide the token through ``HF_TOKEN`` (or a prior ``hf auth login``).
In a container, pass ``HF_TOKEN`` as a secret and mount a warm
``HF_HOME`` cache so a cold replica does not re-fetch several gigabytes.

Operational notes
-----------------
*   **The reference transcript is normalised too.** It is passed through
    :func:`mlvoice.text.unicode_norm.normalize` before reaching the model, so
    that the reference and the target are in the same orthographic form. A
    transcript written with the legacy ``ൻറ`` spelling against a target
    normalised to ``ന്റ`` would otherwise present the model with two spellings
    of the same word, one of which it may never have seen.
*   **The reference transcript matters.** The model conditions on ``ref_text``;
    supplying the wrong transcript degrades the clone badly. Enrolment stores a
    verified transcript for exactly this reason.
*   **transformers must be older than 4.51.** The bundled code constructs its
    vocoder inside ``__init__`` and moves it to the device, which cannot work
    if the parameters were allocated on the meta device. Up to 4.50.3,
    ``low_cpu_mem_usage=False`` (which ``load`` passes) turns meta
    initialisation off. In 4.51.0 that switch was replaced by
    ``get_init_context()`` and meta initialisation became unconditional, so the
    flag is ignored and the model cannot load at all -- it fails with "Cannot
    copy out of meta tensor". The ``indicf5`` extra pins ``<4.51`` for this
    reason; it is not a precaution, it is the difference between loading and
    not.
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

import re
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
                # Recent transformers builds the model on the meta device by
                # default, materialising weights afterwards. That is fine for a
                # plain nn.Module, but IndicF5's remote code constructs its
                # vocos vocoder inside __init__ and then calls .to(device) on
                # it. Under meta init the vocoder's parameters have no storage,
                # so the move fails with "Cannot copy out of meta tensor".
                # Disabling the fast path makes the constructor allocate real
                # tensors, which is what the model's code assumes.
                low_cpu_mem_usage=False,
            )
            self._model = model.to(torch.device(self._device)).eval()
        except Exception as exc:
            reason = str(exc)
            hint = _load_failure_hint(reason, self._model_id)
            raise BackendUnavailableError(
                "could not load the IndicF5 weights",
                model_id=self._model_id,
                revision=self._revision,
                reason=reason,
                **({"hint": hint} if hint else {}),
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


_MISSING_DEPS: Final = re.compile(
    r"requires the following packages that were not found[^:]*:\s*([^\n.]+)"
)


def _load_failure_hint(reason: str, model_id: str) -> str | None:
    """Turn a load failure into an actionable instruction, where one exists.

    Two failures account for nearly every first run, and neither message says
    what to do:

    * the repository is gated, and the caller sees a bare 401; and
    * the model's bundled remote code imports packages that are not declared
      by this project's dependencies, because they are the model's
      requirements rather than ours.

    Anything else gets no hint, rather than a misleading one.
    """
    lowered = reason.lower()
    if "401" in reason or "gated" in lowered or "restricted" in lowered:
        return (
            f"{model_id} is a gated repository: request access on its model "
            "page, then authenticate with `hf auth login` or set HF_TOKEN for "
            "the account that was granted access"
        )
    if "meta tensor" in lowered or "to_empty" in lowered:
        return (
            "the model's remote code builds its vocoder during __init__ and "
            "moves it to the device, which fails when transformers allocates "
            "on the meta device. transformers 4.51.0 made that unconditional "
            "and ignores low_cpu_mem_usage, so this needs an older release: "
            "`pip install 'transformers>=4.44,<4.51'`"
        )
    if "load_model()" in reason and "ckpt_path" in reason:
        return (
            "the installed f5_tts is SWivid's upstream package from PyPI, not "
            "AI4Bharat's fork that this model's code targets: the two share an "
            "import name but not an API. Install the fork with "
            "`pip install 'mlvoice[indicf5]'`, which replaces it"
        )
    match = _MISSING_DEPS.search(reason)
    if match:
        packages = ", ".join(p.strip() for p in match.group(1).split(",") if p.strip())
        return (
            f"the model's remote code imports {packages}, which are its own "
            "requirements rather than this project's: install them with "
            "`pip install 'mlvoice[indicf5]'`"
        )
    return None


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
