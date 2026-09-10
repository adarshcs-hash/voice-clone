"""Cross-cutting protocols.

Three capabilities are needed by more than one layer: ASR (corpus verification,
consent verification, evaluation), speaker verification (consent, evaluation)
and perceptual quality estimation (corpus filtering, evaluation).

They live here, at the bottom of the dependency graph, rather than in whichever
package happened to need them first. That keeps the layers acyclic: ``voices``
and ``eval`` both depend on these protocols and neither depends on the other.

Concrete implementations live with the layer that owns the model --
:class:`mlvoice.eval.speaker.EcapaSpeakerVerifier`, for instance -- so that
importing a protocol never pulls in a model runtime.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mlvoice.audio.io import Audio

__all__ = ["MosEstimator", "SpeakerVerifier", "Transcriber"]


@runtime_checkable
class Transcriber(Protocol):
    """Speech recognition.

    Used to verify that a transcript matches its audio (corpus preparation),
    that a consent phrase was actually spoken (enrolment), and to measure
    intelligibility by round-trip (evaluation).

    For Malayalam, an Indic-specific recogniser (IndicConformer, IndicWhisper)
    is materially better than general multilingual Whisper. A weak recogniser
    here does active harm: in corpus preparation it rejects good data and keeps
    bad, and in evaluation it reports error that belongs to itself.
    """

    def transcribe(self, audio: Audio) -> str:
        """Return the recognised text for ``audio``."""
        ...


@runtime_checkable
class SpeakerVerifier(Protocol):
    """Speaker similarity scoring.

    Scores are not comparable across models; calibrate any threshold against
    same-speaker and different-speaker pairs from your own data.
    """

    def similarity(self, first: Audio, second: Audio) -> float:
        """Cosine similarity of speaker embeddings, in ``[-1, 1]``."""
        ...

    @property
    def name(self) -> str:
        """Identifier recorded in reports and audit records."""
        ...


@runtime_checkable
class MosEstimator(Protocol):
    """No-reference perceptual quality estimation (UTMOS, DNSMOS)."""

    def score(self, audio: Audio) -> float:
        """Predicted mean opinion score, nominally 1-5."""
        ...
