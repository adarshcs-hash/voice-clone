"""Objective metrics for Malayalam TTS.

Why CER and not WER
-------------------
Malayalam is agglutinative: a single orthographic word can carry what English
spreads over five or six (``വീട്ടിലേക്കായിരുന്നു`` -- "it was towards the
house"). One wrong morpheme therefore fails an entire long word, so WER is both
coarse and unstable across systems that segment differently. Character error
rate is the metric that moves proportionally to what a listener notices, and it
is what this project gates on. WER is reported alongside it for comparability
with published numbers, never as the primary signal.

Metrics implemented here are dependency-free. Where a reference implementation
exists (``jiwer``), the test suite cross-checks against it rather than the code
depending on it, so that CI on a bare box still measures the thing it gates on.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Final

from mlvoice.errors import ValidationError

__all__ = [
    "ErrorRate",
    "character_error_rate",
    "edit_distance",
    "normalize_for_scoring",
    "word_error_rate",
]

_PUNCTUATION_CATEGORIES: Final[frozenset[str]] = frozenset({"Po", "Pd", "Ps", "Pe", "Pi", "Pf"})


@dataclass(frozen=True, slots=True)
class ErrorRate:
    """An error rate with the counts that produced it."""

    rate: float
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def errors(self) -> int:
        """Total edit operations."""
        return self.substitutions + self.deletions + self.insertions

    def as_dict(self) -> dict[str, float | int]:
        """Flat mapping, for report serialisation."""
        return {
            "rate": round(self.rate, 6),
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "reference_length": self.reference_length,
        }


def normalize_for_scoring(text: str, *, strip_punctuation: bool = True) -> str:
    """Canonicalise text before scoring.

    Applies Malayalam Unicode normalisation so that a chillu spelling difference
    is not counted as an error, folds whitespace, and optionally removes
    punctuation -- an ASR system's comma placement is not the voice's fault.
    """
    from mlvoice.text.unicode_norm import normalize

    text = normalize(text)
    if strip_punctuation:
        text = "".join(ch for ch in text if unicodedata.category(ch) not in _PUNCTUATION_CATEGORIES)
    return " ".join(text.split())


def edit_distance(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """Levenshtein alignment counts between two token sequences.

    Uses the standard dynamic-programming table with backtracking over
    operation classes, in ``O(len(reference) * len(hypothesis))`` time and
    ``O(len(hypothesis))`` space for the cost row plus a full table for the
    backtrace.

    Returns:
        ``(substitutions, deletions, insertions)``.
    """
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return 0, 0, m
    if m == 0:
        return 0, n, 0

    # table[i][j] = (cost, substitutions, deletions, insertions)
    table: list[list[tuple[int, int, int, int]]] = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        table[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        table[i][0] = (i, 0, i, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cost, sub, dele, ins = table[i - 1][j - 1]
                table[i][j] = (cost, sub, dele, ins)
                continue
            sub_cell = table[i - 1][j - 1]
            del_cell = table[i - 1][j]
            ins_cell = table[i][j - 1]
            best = min(sub_cell[0], del_cell[0], ins_cell[0]) + 1
            if sub_cell[0] <= del_cell[0] and sub_cell[0] <= ins_cell[0]:
                table[i][j] = (best, sub_cell[1] + 1, sub_cell[2], sub_cell[3])
            elif del_cell[0] <= ins_cell[0]:
                table[i][j] = (best, del_cell[1], del_cell[2] + 1, del_cell[3])
            else:
                table[i][j] = (best, ins_cell[1], ins_cell[2], ins_cell[3] + 1)

    _, substitutions, deletions, insertions = table[n][m]
    return substitutions, deletions, insertions


def _rate(reference: list[str], hypothesis: list[str]) -> ErrorRate:
    if not reference:
        raise ValidationError("cannot score against an empty reference")
    substitutions, deletions, insertions = edit_distance(reference, hypothesis)
    errors = substitutions + deletions + insertions
    return ErrorRate(
        rate=errors / len(reference),
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_length=len(reference),
    )


def character_error_rate(
    reference: str, hypothesis: str, *, strip_punctuation: bool = True
) -> ErrorRate:
    """Character error rate, the primary quality gate for Malayalam.

    Whitespace is removed before comparison so that a tokenisation difference
    does not register as an error.

    Raises:
        ValidationError: ``reference`` normalises to the empty string.

    Examples:
        >>> character_error_rate("നാട്", "നാട്").rate
        0.0
    """
    ref = normalize_for_scoring(reference, strip_punctuation=strip_punctuation).replace(" ", "")
    hyp = normalize_for_scoring(hypothesis, strip_punctuation=strip_punctuation).replace(" ", "")
    return _rate(list(ref), list(hyp))


def word_error_rate(
    reference: str, hypothesis: str, *, strip_punctuation: bool = True
) -> ErrorRate:
    """Word error rate. Reported for comparability; not the primary gate.

    Raises:
        ValidationError: ``reference`` normalises to the empty string.
    """
    ref = normalize_for_scoring(reference, strip_punctuation=strip_punctuation).split()
    hyp = normalize_for_scoring(hypothesis, strip_punctuation=strip_punctuation).split()
    return _rate(ref, hyp)
