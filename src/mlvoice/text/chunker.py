"""Sentence and prosodic chunking.

Autoregressive TTS degrades on long inputs: prosody drifts, and the risk of a
repetition or truncation failure grows with sequence length. Long-form
synthesis is therefore done chunk by chunk, and *where* the chunks fall decides
how natural the result sounds -- a break mid-clause is audible.

The chunker splits on sentence terminators first, then on clause-level
punctuation, and only as a last resort on a word boundary. Malayalam is
agglutinative, so single words can exceed 30 characters and a naive
fixed-window split will cut inside one; :func:`chunk` never does.

Chunks carry the pause that should follow them, so the synthesis layer can
insert silence rather than relying on the acoustic model to infer a break it
cannot see.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from mlvoice.errors import ValidationError

__all__ = ["BreakStrength", "Chunk", "ChunkConfig", "chunk"]

# Sentence terminators, including the danda that survives in some Malayalam copy.
_SENTENCE_END: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?।॥])\s+")
# Clause-level breaks, used when a sentence is still too long.
_CLAUSE_END: Final[re.Pattern[str]] = re.compile(r"(?<=[,;:—-])\s+")
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


class BreakStrength(StrEnum):
    """How strong a pause should follow a chunk."""

    NONE = "none"
    WORD = "word"
    CLAUSE = "clause"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


_PAUSE_MS: Final[dict[BreakStrength, int]] = {
    BreakStrength.NONE: 0,
    BreakStrength.WORD: 60,
    BreakStrength.CLAUSE: 180,
    BreakStrength.SENTENCE: 380,
    BreakStrength.PARAGRAPH: 700,
}


@dataclass(frozen=True, slots=True)
class Chunk:
    """One synthesis unit."""

    text: str
    break_after: BreakStrength
    index: int

    @property
    def pause_ms(self) -> int:
        """Silence to insert after this chunk, in milliseconds."""
        return _PAUSE_MS[self.break_after]


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Chunking bounds.

    Attributes:
        max_chars: Hard upper bound per chunk. Chosen to sit inside the
            sequence length the acoustic model was trained on.
        min_chars: Chunks shorter than this are merged forward where possible,
            so that a stray two-word fragment does not get its own generation
            pass with its own prosody.
    """

    max_chars: int = 220
    min_chars: int = 40

    def __post_init__(self) -> None:
        if self.max_chars < 1:
            raise ValidationError("max_chars must be positive", max_chars=self.max_chars)
        if self.min_chars >= self.max_chars:
            raise ValidationError(
                "min_chars must be smaller than max_chars",
                min_chars=self.min_chars,
                max_chars=self.max_chars,
            )


DEFAULT_CONFIG: Final = ChunkConfig()


def _split_words(text: str, max_chars: int) -> list[str]:
    """Split on word boundaries, packing greedily up to ``max_chars``.

    A single word longer than ``max_chars`` is emitted whole rather than cut:
    an agglutinated Malayalam word is one prosodic unit, and splitting it
    produces two mispronounced fragments.
    """
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for word in _WHITESPACE.split(text.strip()):
        if not word:
            continue
        addition = len(word) + (1 if current else 0)
        if current and size + addition > max_chars:
            pieces.append(" ".join(current))
            current, size = [word], len(word)
        else:
            current.append(word)
            size += addition
    if current:
        pieces.append(" ".join(current))
    return pieces


def _split_to_limit(
    text: str, max_chars: int, strength: BreakStrength
) -> list[tuple[str, BreakStrength]]:
    """Recursively split ``text`` until every piece fits ``max_chars``."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [(text, strength)]

    clauses = _CLAUSE_END.split(text)
    if len(clauses) > 1:
        out: list[tuple[str, BreakStrength]] = []
        for i, clause in enumerate(clauses):
            last = i == len(clauses) - 1
            out.extend(
                _split_to_limit(clause, max_chars, strength if last else BreakStrength.CLAUSE)
            )
        return out

    words = _split_words(text, max_chars)
    return [
        (piece, strength if i == len(words) - 1 else BreakStrength.WORD)
        for i, piece in enumerate(words)
    ]


def _merge_short(
    pieces: list[tuple[str, BreakStrength]], config: ChunkConfig
) -> list[tuple[str, BreakStrength]]:
    """Merge a too-short piece into the next one when the result still fits."""
    merged: list[tuple[str, BreakStrength]] = []
    for text, strength in pieces:
        if (
            merged
            and len(merged[-1][0]) < config.min_chars
            and len(merged[-1][0]) + 1 + len(text) <= config.max_chars
        ):
            prev_text, _ = merged.pop()
            merged.append((f"{prev_text} {text}", strength))
        else:
            merged.append((text, strength))
    return merged


def chunk(text: str, config: ChunkConfig = DEFAULT_CONFIG) -> list[Chunk]:
    """Split ``text`` into synthesis chunks with the pause that follows each.

    Splitting is attempted at progressively weaker boundaries: paragraph,
    sentence, clause, word. A word is never split.

    Args:
        text: Fully expanded, normalised text.
        config: Chunk size bounds.

    Returns:
        Chunks in reading order. The final chunk always carries
        :attr:`BreakStrength.SENTENCE` or weaker, never a paragraph break.

    Examples:
        >>> [c.text for c in chunk("ഒന്ന്. രണ്ട്.")]
        ['ഒന്ന്.', 'രണ്ട്.']
    """
    pieces: list[tuple[str, BreakStrength]] = []
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    for p_index, paragraph in enumerate(paragraphs):
        sentences = [s for s in _SENTENCE_END.split(paragraph.strip()) if s.strip()]
        for s_index, sentence in enumerate(sentences):
            last_sentence = s_index == len(sentences) - 1
            last_paragraph = p_index == len(paragraphs) - 1
            if last_sentence and last_paragraph:
                tail = BreakStrength.NONE
            elif last_sentence:
                tail = BreakStrength.PARAGRAPH
            else:
                tail = BreakStrength.SENTENCE
            pieces.extend(_split_to_limit(sentence, config.max_chars, tail))

    pieces = _merge_short(pieces, config)
    return [
        Chunk(text=text_piece, break_after=strength, index=i)
        for i, (text_piece, strength) in enumerate(pieces)
    ]
