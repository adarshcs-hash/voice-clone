"""Speaker similarity.

Speaker similarity is how a *cloning* system is graded: a voice can be natural,
intelligible and correctly pronounced while sounding like somebody else. The
metric is the cosine similarity of speaker embeddings from a verification model
trained for exactly that discrimination.

Only a real model is offered. A cheap spectral-distance proxy would produce a
number that correlates with recording conditions rather than with speaker
identity, and a misleading metric is worse than a missing one: it gets tracked,
optimised against, and quietly steers the project wrong.

Calibrate before trusting a threshold. Cosine scores are not comparable across
models, and the same 0.7 that means "same speaker" for one checkpoint can mean
"unrelated" for another. Score a few hundred same-speaker and different-speaker
pairs from your own corpus, and set the threshold from that distribution.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mlvoice.audio.io import Audio, resample
from mlvoice.errors import BackendUnavailableError
from mlvoice.logging import get_logger

__all__ = ["EcapaSpeakerVerifier"]

# Implements mlvoice.protocols.SpeakerVerifier.

log = get_logger(__name__)


class EcapaSpeakerVerifier:
    """ECAPA-TDNN speaker verification via SpeechBrain.

    Args:
        model_id: SpeechBrain model source.
        device: ``cpu`` or ``cuda``.

    Requires the ``speaker`` extra.
    """

    _MODEL_RATE = 16_000

    def __init__(
        self,
        *,
        model_id: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: str = "cpu",
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._model: Any | None = None

    @property
    def name(self) -> str:
        """Identifier recorded in reports and audit records."""
        return f"ecapa:{self._model_id}"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            raise BackendUnavailableError(
                "EcapaSpeakerVerifier requires the 'eval' extra: pip install 'mlvoice[eval]'"
            ) from exc
        self._model = EncoderClassifier.from_hparams(
            source=self._model_id, run_opts={"device": self._device}
        )
        log.info("speaker verifier loaded", model_id=self._model_id, device=self._device)

    def embed(self, audio: Audio) -> np.ndarray:
        """Return the L2-normalised speaker embedding for ``audio``."""
        self._ensure_loaded()
        import torch

        assert self._model is not None
        working = resample(audio, self._MODEL_RATE)
        with torch.inference_mode():
            tensor = torch.from_numpy(working.samples).unsqueeze(0)
            embedding = self._model.encode_batch(tensor).squeeze().cpu().numpy()
        vector = np.asarray(embedding, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def similarity(self, first: Audio, second: Audio) -> float:
        """Cosine similarity of the two recordings' speaker embeddings."""
        return float(np.dot(self.embed(first), self.embed(second)))
