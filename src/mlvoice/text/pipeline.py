"""The Malayalam text frontend, assembled.

One entry point, :meth:`TextPipeline.process`, runs the stages in the only order
that is correct:

1.  **Normalise** Unicode, so that every later stage sees one spelling per word
    and Malayalam digits have become ASCII.
2.  **Expand** notations -- numbers, dates, times, currency, units,
    abbreviations -- while the digits are still digits.
3.  **Route code-mix**, turning Manglish into Malayalam script and acronyms into
    spelled-out Malayalam, after expansion has consumed the digits an acronym
    rule might otherwise mangle.
4.  **Chunk** into synthesis units at the strongest available boundary.
5.  **Phonemise** each chunk.

Every stage's output is retained on :class:`ProcessedText`. That costs a little
memory and buys the ability to answer "why did the voice say that?" from a
single log line in production, which is the question that actually gets asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from mlvoice.errors import TextTooLongError, ValidationError
from mlvoice.text.chars import has_malayalam
from mlvoice.text.chunker import Chunk, ChunkConfig, chunk
from mlvoice.text.codemix import CodeMixConfig, route
from mlvoice.text.expand import ExpansionConfig, expand
from mlvoice.text.g2p import G2PConfig, Notation, phonemize, to_string
from mlvoice.text.unicode_norm import NormalizationReport, normalize_with_report

__all__ = ["ProcessedText", "TextPipeline", "TextPipelineConfig"]


@dataclass(frozen=True, slots=True)
class TextPipelineConfig:
    """Composed configuration for the whole frontend."""

    max_chars: int = 5_000
    aggressive_legacy_normalization: bool = False
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)
    codemix: CodeMixConfig = field(default_factory=CodeMixConfig)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    g2p: G2PConfig = field(default_factory=G2PConfig)
    notation: Notation = Notation.IPA


DEFAULT_CONFIG: Final = TextPipelineConfig()


@dataclass(frozen=True, slots=True)
class ProcessedText:
    """The frontend's output, with every intermediate stage retained."""

    original: str
    normalized: str
    expanded: str
    routed: str
    chunks: tuple[Chunk, ...]
    phonemes: tuple[tuple[tuple[str, ...], ...], ...]
    """Per chunk, per word, the phoneme tokens."""
    normalization: NormalizationReport
    contains_malayalam: bool

    @property
    def text_for_synthesis(self) -> str:
        """The graphemic text a grapheme-input acoustic model should receive."""
        return self.routed

    def phoneme_string(self, chunk_index: int, *, notation: Notation = Notation.IPA) -> str:
        """Render one chunk's phonemes as a single string.

        Raises:
            ValidationError: ``chunk_index`` is out of range.
        """
        if not 0 <= chunk_index < len(self.phonemes):
            raise ValidationError(
                "chunk index out of range",
                chunk_index=chunk_index,
                chunk_count=len(self.phonemes),
            )
        words = [list(word) for word in self.phonemes[chunk_index]]
        return to_string(words, notation=notation)

    def summary(self) -> dict[str, object]:
        """Compact, log-safe description of what the frontend did."""
        return {
            "chars_in": len(self.original),
            "chars_out": len(self.routed),
            "chunks": len(self.chunks),
            "normalization_changes": self.normalization.total_changes,
            "legacy_nta_fixed": self.normalization.legacy_nta_fixed,
            "contains_malayalam": self.contains_malayalam,
        }


class TextPipeline:
    """Stateless, thread-safe Malayalam text frontend.

    Holds only immutable configuration, so a single instance can be shared
    across request handlers and worker threads.

    Examples:
        >>> pipeline = TextPipeline()
        >>> result = pipeline.process("₹250 ഉണ്ട്")
        >>> result.routed
        'ഇരുനൂറ്റിയമ്പത് രൂപ ഉണ്ട്'
    """

    def __init__(self, config: TextPipelineConfig = DEFAULT_CONFIG) -> None:
        self._config = config

    @property
    def config(self) -> TextPipelineConfig:
        """The configuration this pipeline was built with."""
        return self._config

    def process(self, text: str) -> ProcessedText:
        """Run the full frontend over ``text``.

        Args:
            text: Raw user input. May mix Malayalam, Latin, digits and legacy
                Malayalam encodings.

        Returns:
            A :class:`ProcessedText` carrying every stage's output.

        Raises:
            ValidationError: ``text`` is empty or whitespace only.
            TextTooLongError: ``text`` exceeds the configured character budget.
        """
        if not text or not text.strip():
            raise ValidationError("text is empty")
        if len(text) > self._config.max_chars:
            raise TextTooLongError(
                "text exceeds the per-request character budget",
                chars=len(text),
                limit=self._config.max_chars,
            )

        normalized, report = normalize_with_report(
            text, aggressive_legacy=self._config.aggressive_legacy_normalization
        )
        expanded = expand(normalized, self._config.expansion)
        routed = route(expanded, self._config.codemix)
        chunks = tuple(chunk(routed, self._config.chunking))
        phonemes = tuple(
            tuple(tuple(word) for word in phonemize(c.text, self._config.g2p)) for c in chunks
        )
        return ProcessedText(
            original=text,
            normalized=normalized,
            expanded=expanded,
            routed=routed,
            chunks=chunks,
            phonemes=phonemes,
            normalization=report,
            contains_malayalam=has_malayalam(routed),
        )
