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
from mlvoice.text.chars import is_consonant, is_independent_vowel

__all__ = ["BreakStrength", "Chunk", "ChunkConfig", "chunk"]

# Candidate sentence boundaries: a terminator followed by whitespace. Whether a
# full stop really ends a sentence is decided by :func:`_is_initial`.
_SENTENCE_BOUNDARY: Final[re.Pattern[str]] = re.compile(r"([.!?।॥])(\s+)")
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
        pack_sentences: Merge consecutive sentences into one chunk while they
            fit within ``max_chars``. On by default, because a reference-prompt
            model pays for its reference clip once per generation: fewer, fuller
            chunks are markedly faster and keep prosody continuous. Turn it off
            for streaming, where a small first chunk means a lower
            time-to-first-audio.
    """

    max_chars: int = 220
    min_chars: int = 40
    pack_sentences: bool = True

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


def _letter_count(token: str) -> int:
    """Count orthographic letters, not codepoints.

    A Malayalam letter is a base consonant or independent vowel plus any vowel
    signs and chandrakkala hanging off it, so ``വി`` is one letter across two
    codepoints while ``ശരി`` is two letters across three. Counting bases is
    what distinguishes an initial from a short word.
    """
    return sum(
        1
        for ch in token
        if is_consonant(ch) or is_independent_vowel(ch) or (ch.isalpha() and ch.isascii())
    )


def _is_initial(text_before_stop: str) -> bool:
    """True if the full stop at the end of ``text_before_stop`` marks an initial.

    ``ഒ. ജെ ജനീഷ്`` and ``വി. ഡി സതീശൻ`` are one name each, not three
    sentences. Initials are near-universal in Malayalam names in news and
    social copy, and splitting at them produces a hard stop mid-name plus a
    fragment that the model then renders with its own sentence prosody.

    The test is a single orthographic letter immediately before the stop.
    That distinguishes ``വി.`` (one letter) from ``ശരി.`` (two), which a
    codepoint count cannot.
    """
    token = text_before_stop.rsplit(maxsplit=1)[-1] if text_before_stop.strip() else ""
    return _letter_count(token) == 1


def _split_sentences(paragraph: str) -> list[str]:
    """Split into sentences, keeping initials attached to their names."""
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(paragraph):
        terminator, end = match.group(1), match.end()
        if terminator == "." and _is_initial(paragraph[start : match.start()]):
            continue
        piece = paragraph[start : match.start() + 1].strip()
        if piece:
            sentences.append(piece)
        start = end
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _pack(
    pieces: list[tuple[str, BreakStrength]], max_chars: int
) -> list[tuple[str, BreakStrength]]:
    """Merge consecutive pieces up to ``max_chars``, stopping at a paragraph.

    A reference-prompt model re-synthesises the reference clip for *every*
    generation and discards it, so each extra chunk costs the reference's
    duration again. Ten short sentences against a ten-second reference throw
    away a hundred seconds of audio. Packing them into as few generations as
    the model handles well is therefore a large speed win, and it keeps
    sentence-to-sentence prosody inside one generation rather than stitching it
    across a seam.

    Paragraph breaks are never packed across: that pause is meaningful, and the
    text either side of it is not one prosodic unit.
    """
    packed: list[tuple[str, BreakStrength]] = []
    for text, strength in pieces:
        if packed:
            previous_text, previous_strength = packed[-1]
            joinable = (
                previous_strength not in (BreakStrength.PARAGRAPH, BreakStrength.WORD)
                and len(previous_text) + 1 + len(text) <= max_chars
            )
            if joinable:
                packed[-1] = (f"{previous_text} {text}", strength)
                continue
        packed.append((text, strength))
    return packed


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

    clauses = _CLAUSE_END.split(text)  # noqa: RUF100
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
        sentences = _split_sentences(paragraph.strip())
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

    if config.pack_sentences:
        pieces = _pack(pieces, config.max_chars)
    pieces = _merge_short(pieces, config)
    return [
        Chunk(text=text_piece, break_after=strength, index=i)
        for i, (text_piece, strength) in enumerate(pieces)
    ]
