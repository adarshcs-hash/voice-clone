"""Speech recognition, for transcribing reference clips.

Why this module exists at all: a reference-prompt TTS model conditions on the
reference clip *and its transcript*, so a product that asks the user to type
that transcript has made its own quality depend on the least reliable step in
the flow. People mistype, paraphrase, or -- most commonly -- paste the text they
want generated. Every one of those clones the voice correctly and garbles the
words, which is the hardest failure to diagnose from the outside.

Transcribing the clip removes the question. The right interaction is to
transcribe, *show* the result, and let the user correct it: ASR is not perfect
either, but a visible approximate transcript beats an invisible wrong one.

Model choice matters
--------------------
For Malayalam, an Indic-specific recogniser is materially better than general
multilingual Whisper. Candidates, best first:

``ai4bharat/indic-conformer-600m-multilingual``
    AI4Bharat's IndicConformer. Trained for Indian languages, Malayalam
    included. Needs ``trust_remote_code``.
``vasista22/whisper-malayalam-medium``
    Malayalam-specific Whisper fine-tune.
``openai/whisper-large-v3``
    Supports Malayalam and is ungated and dependable, but noticeably weaker on
    it than the above. A reasonable default when nothing else is configured.
``openai/whisper-small``
    Fast, and weak enough on Malayalam that it should be treated as a smoke
    test rather than a transcriber.

The model id is configuration, not code, because which of these is best will
change and because the right answer depends on whether accuracy or latency
matters more for a given deployment.
"""

from __future__ import annotations

from typing import Any, Final

from mlvoice.audio.io import Audio, resample
from mlvoice.errors import BackendUnavailableError
from mlvoice.logging import get_logger

__all__ = ["ASR_MODEL_SAMPLE_RATE", "TransformersTranscriber"]

log = get_logger(__name__)

ASR_MODEL_SAMPLE_RATE: Final = 16_000
"""Whisper and the wav2vec2/conformer families all expect 16 kHz."""


class TransformersTranscriber:
    """Speech recognition through the ``transformers`` ASR pipeline.

    Implements :class:`mlvoice.protocols.Transcriber`. Loads lazily and once,
    because model construction dominates the cost of a single transcription and
    a service transcribes on every enrolment.

    Args:
        model_id: Hub repository id. See the module docstring for candidates.
        revision: Commit to pin. Recommended for the same reason it is for the
            synthesis model, and required when ``trust_remote_code`` is set.
        device: ``cpu``, ``cuda`` or ``mps``.
        language: Language hint, where the model accepts one. Whisper will
            otherwise detect the language, and it detects Malayalam
            unreliably on short clips -- which reference clips always are.
        trust_remote_code: Needed by IndicConformer and models like it.
    """

    def __init__(
        self,
        *,
        model_id: str = "openai/whisper-large-v3",
        revision: str | None = None,
        device: str = "cpu",
        language: str | None = "ml",
        trust_remote_code: bool = False,
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._device = device
        self._language = language
        self._trust_remote_code = trust_remote_code
        self._pipeline: Any | None = None

    @property
    def name(self) -> str:
        """Identifier recorded in reports and audit records."""
        return f"transformers-asr:{self._model_id}"

    def is_ready(self) -> bool:
        """True once the model is resident."""
        return self._pipeline is not None

    def load(self) -> None:
        """Load the recogniser. Idempotent.

        Raises:
            BackendUnavailableError: The ``models`` extra is missing, or the
                model cannot be loaded.
        """
        if self._pipeline is not None:
            return
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise BackendUnavailableError(
                "transcription requires the 'models' extra: pip install 'mlvoice[models]'"
            ) from exc

        try:
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self._model_id,
                revision=self._revision,
                device=torch.device(self._device),
                trust_remote_code=self._trust_remote_code,
            )
        except Exception as exc:
            raise BackendUnavailableError(
                "could not load the transcription model",
                model_id=self._model_id,
                revision=self._revision,
                reason=str(exc),
            ) from exc
        log.info(
            "transcriber loaded",
            model_id=self._model_id,
            revision=self._revision,
            device=self._device,
        )

    def transcribe(self, audio: Audio) -> str:
        """Return the recognised text for ``audio``.

        Raises:
            BackendUnavailableError: The model is not loaded and cannot be.
        """
        self.load()
        assert self._pipeline is not None

        working = resample(audio, ASR_MODEL_SAMPLE_RATE)
        kwargs: dict[str, Any] = {}
        if self._language is not None:
            # Whisper takes the hint through generate_kwargs; models that do not
            # accept it raise, so a failure here falls back to detection rather
            # than failing the request.
            kwargs["generate_kwargs"] = {"language": self._language}

        payload = {"raw": working.samples, "sampling_rate": working.sample_rate}
        try:
            result = self._pipeline(payload, **kwargs)
        except (TypeError, ValueError) as exc:
            log.warning(
                "transcriber rejected the language hint; retrying with detection",
                model_id=self._model_id,
                reason=str(exc),
            )
            result = self._pipeline(payload)

        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return str(text).strip()
