"""Malayalam text frontend.

The public surface is :class:`~mlvoice.text.pipeline.TextPipeline`; the
individual stages are exported for targeted use (corpus cleaning, lexicon work,
evaluation) and are individually testable.
"""

from mlvoice.text.chunker import BreakStrength, Chunk, ChunkConfig, chunk
from mlvoice.text.codemix import (
    CodeMixConfig,
    LatinPolicy,
    ManglishDetection,
    route,
    segment,
)
from mlvoice.text.expand import ExpansionConfig, expand
from mlvoice.text.g2p import G2PConfig, Notation, phonemize, phonemize_word, to_string
from mlvoice.text.pipeline import ProcessedText, TextPipeline, TextPipelineConfig
from mlvoice.text.translit import transliterate
from mlvoice.text.unicode_norm import NormalizationReport, normalize, normalize_with_report

__all__ = [
    "BreakStrength",
    "Chunk",
    "ChunkConfig",
    "CodeMixConfig",
    "ExpansionConfig",
    "G2PConfig",
    "LatinPolicy",
    "ManglishDetection",
    "NormalizationReport",
    "Notation",
    "ProcessedText",
    "TextPipeline",
    "TextPipelineConfig",
    "chunk",
    "expand",
    "normalize",
    "normalize_with_report",
    "phonemize",
    "phonemize_word",
    "route",
    "segment",
    "to_string",
    "transliterate",
]
