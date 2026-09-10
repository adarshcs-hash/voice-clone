"""mlvoice — Malayalam text-to-speech and voice cloning.

The package is layered so that the parts which carry the most product value
(the Malayalam text frontend and the evaluation harness) have no dependency on
model runtimes and can be developed, tested and shipped independently of GPUs.

Layers, bottom up:

``mlvoice.text``     Deterministic Malayalam text processing: Unicode
                     normalisation, number/date expansion, transliteration,
                     code-mix routing and grapheme-to-phoneme conversion.
``mlvoice.audio``    Audio IO, loudness normalisation and quality gates.
``mlvoice.data``     Corpus manifests and the preparation pipeline.
``mlvoice.tts``      Synthesiser protocol and concrete backends.
``mlvoice.voices``   Voice enrolment, consent records and persistence.
``mlvoice.safety``   Watermarking and moderation.
``mlvoice.eval``     Objective metrics and the Malayalam hard test set.
``mlvoice.api``      HTTP service.
"""

from mlvoice.__version__ import __version__

__all__ = ["__version__"]
