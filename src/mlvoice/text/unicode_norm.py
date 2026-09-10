"""Malayalam Unicode normalisation.

Malayalam is written with a script that admits several byte sequences for the
same word. Left unnormalised, a TTS tokenizer sees them as different tokens,
splitting training signal and producing mispronunciations at inference. The
recurring sources of variation are:

1.  **Chillu spelling.** A pure consonant may be an atomic chillu (``ൻ``
    U+0D7B) or the sequence base + virama + ZWJ (``ന`` + ``്`` + ZWJ). Both
    render identically.
2.  **Legacy ``nta``.** ASCII-to-Unicode converters and older keyboards emit
    ``ൻ`` + ``റ`` for what should be ``ന`` + virama + ``റ`` (``ന്റ``). This is
    by far the most common real-world defect in Malayalam corpora.
3.  **The AU sign.** ``ൌ`` (U+0D4C) and ``ൗ`` (U+0D57) are visually the same;
    modern orthography uses the latter.
4.  **Dot reph.** ``ൎ`` (U+0D4E) is a historic superscript /r/ encoded in
    logical position, equivalent to ``ർ``.
5.  **Zero-width controls.** ZWNJ only affects glyph selection and carries no
    phonetic information; stray ZWJ not forming a chillu is noise.
6.  **Malayalam digits and numerals**, which downstream number expansion does
    not handle and which must become ASCII digits first.

:func:`normalize` applies these in a fixed order and can report what it
changed, which the corpus pipeline uses to quantify how dirty a source is.

Ordering matters: ZWJ-based chillu must be composed *before* stray zero-width
controls are stripped, and the legacy ``nta`` repair must run *after* chillu
composition so that both spellings of the defect are caught.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Final

from mlvoice.text.chars import (
    ANUSVARA,
    ARCHAIC_II,
    AU_LENGTH_MARK,
    BASE_TO_CHILLU,
    CHILLU_LLL,
    CHILLU_M,
    CHILLU_N,
    CHILLU_NN,
    CHILLU_RR,
    CHILLU_Y,
    DOT_REPH,
    MALAYALAM_DIGITS,
    MALAYALAM_NUMBER_SIGNS,
    VIRAMA,
    VOWEL_SIGN_AU,
    ZWJ,
    ZWNJ,
)

__all__ = ["NormalizationReport", "normalize", "normalize_with_report"]

# Composition: base + virama + ZWJ -> atomic chillu.
_CHILLU_COMPOSE: Final[re.Pattern[str]] = re.compile(
    "([" + "".join(BASE_TO_CHILLU) + "])" + re.escape(VIRAMA) + ZWJ
)

# Legacy nta: atomic chillu-n immediately followed by ``റ``.
_LEGACY_NTA: Final[re.Pattern[str]] = re.compile(CHILLU_N + "റ")

# Optional, more aggressive legacy repairs. These are *not* applied by default
# because each has legitimate modern counter-examples across a compound
# boundary (``മുൻനിര`` "front row" really is chillu-n + na).
_AGGRESSIVE_LEGACY: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(CHILLU_N + "ന"), "ന" + VIRAMA + "ന"),
    (re.compile(CHILLU_N + "മ"), "ന" + VIRAMA + "മ"),
    (re.compile(CHILLU_NN + "ട"), "ണ" + VIRAMA + "ട"),
    (re.compile(CHILLU_NN + "മ"), "ണ" + VIRAMA + "മ"),
)

# Archaic letters with an unambiguous modern equivalent.
_ARCHAIC_MAP: Final[dict[str, str]] = {
    ARCHAIC_II: "ഈ",
    CHILLU_M: ANUSVARA,  # word-final /m/ is written ം in modern orthography
    CHILLU_Y: "യ" + VIRAMA,
    CHILLU_LLL: "ഴ" + VIRAMA,
    DOT_REPH: CHILLU_RR,
    VOWEL_SIGN_AU: AU_LENGTH_MARK,
}

_DANDA_MAP: Final[dict[str, str]] = {"।": ".", "॥": "."}

# Quotes and dashes that punctuation-sensitive chunking should see as ASCII.
_PUNCT_MAP: Final[dict[str, str]] = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "…": "...",
    " ": " ",
    "​": "",  # zero-width space
    "﻿": "",  # BOM
}

_REPEATED_VIRAMA: Final[re.Pattern[str]] = re.compile(f"{re.escape(VIRAMA)}{{2,}}")
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"[^\S\n]+")
_BLANK_LINES: Final[re.Pattern[str]] = re.compile(r"\n{3,}")
_CONTROL: Final[re.Pattern[str]] = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


@dataclass(slots=True)
class NormalizationReport:
    """What :func:`normalize_with_report` changed.

    Counts are the number of substitutions applied, not the number of affected
    words. The corpus pipeline aggregates these per source to decide whether a
    dataset needs a bespoke cleanup pass before training.
    """

    chillu_composed: int = 0
    legacy_nta_fixed: int = 0
    aggressive_legacy_fixed: int = 0
    archaic_mapped: int = 0
    zero_width_stripped: int = 0
    digits_converted: int = 0
    control_stripped: int = 0
    changed: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        """Total substitutions applied."""
        return (
            self.chillu_composed
            + self.legacy_nta_fixed
            + self.aggressive_legacy_fixed
            + self.archaic_mapped
            + self.zero_width_stripped
            + self.digits_converted
            + self.control_stripped
        )


def normalize(
    text: str,
    *,
    aggressive_legacy: bool = False,
    convert_digits: bool = True,
) -> str:
    """Return ``text`` in canonical Malayalam form.

    Args:
        text: Arbitrary input text; may mix Malayalam, Latin and digits.
        aggressive_legacy: Also repair chillu + consonant sequences that are
            *usually* legacy mis-encodings but have legitimate compound-word
            counter-examples. Safe for single-source corpora known to be
            legacy-encoded; unsafe as a default for user input.
        convert_digits: Convert Malayalam digits and numeral signs to ASCII.

    Returns:
        The normalised text. Idempotent: ``normalize(normalize(x)) == normalize(x)``.
    """
    return normalize_with_report(
        text, aggressive_legacy=aggressive_legacy, convert_digits=convert_digits
    )[0]


def normalize_with_report(
    text: str,
    *,
    aggressive_legacy: bool = False,
    convert_digits: bool = True,
) -> tuple[str, NormalizationReport]:
    """Like :func:`normalize`, but also return a :class:`NormalizationReport`."""
    report = NormalizationReport()
    original = text

    # 1. Canonical composition first, so that codepoint-level rules below see a
    #    single representation of each cluster.
    text = unicodedata.normalize("NFC", text)

    # 2. Strip control characters (except newline and tab, handled later).
    text, count = _CONTROL.subn("", text)
    report.control_stripped = count

    # 3. Compose ZWJ-based chillu into atomic chillu.
    text, count = _CHILLU_COMPOSE.subn(lambda m: BASE_TO_CHILLU[m.group(1)], text)
    report.chillu_composed = count

    # 4. Repair the legacy ``nta`` encoding.
    text, count = _LEGACY_NTA.subn("ന" + VIRAMA + "റ", text)
    report.legacy_nta_fixed = count

    if aggressive_legacy:
        for pattern, replacement in _AGGRESSIVE_LEGACY:
            text, count = pattern.subn(replacement, text)
            report.aggressive_legacy_fixed += count

    # 5. Archaic letters -> modern equivalents.
    for src, dst in _ARCHAIC_MAP.items():
        if src in text:
            report.archaic_mapped += text.count(src)
            text = text.replace(src, dst)

    # 6. Remaining zero-width controls carry no phonetic information.
    for zw in (ZWNJ, ZWJ):
        if zw in text:
            report.zero_width_stripped += text.count(zw)
            text = text.replace(zw, "")

    # 7. Malayalam digits and numeral signs -> ASCII.
    if convert_digits:
        for src, dst in (*MALAYALAM_DIGITS.items(), *MALAYALAM_NUMBER_SIGNS.items()):
            if src in text:
                report.digits_converted += text.count(src)
                text = text.replace(src, dst)

    # 8. Punctuation folding.
    for src, dst in (*_DANDA_MAP.items(), *_PUNCT_MAP.items()):
        if src in text:
            text = text.replace(src, dst)

    # 9. Collapse a doubled chandrakkala, which is always a typo.
    text = _REPEATED_VIRAMA.sub(VIRAMA, text)

    # 10. Whitespace: collapse runs, keep at most one blank line, trim.
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n")).strip()

    report.changed = text != original
    if report.legacy_nta_fixed:
        report.notes.append(
            f"repaired {report.legacy_nta_fixed} legacy nta sequence(s) "
            "(chillu-n + ṟa -> na + virama + ṟa)"
        )
    if report.chillu_composed:
        report.notes.append(
            f"composed {report.chillu_composed} ZWJ chillu sequence(s) into atomic chillu"
        )
    return text, report


def decompose_chillu(text: str) -> str:
    """Expand atomic chillu into base + virama + ZWJ.

    The inverse of step 3 in :func:`normalize`. Needed when handing text to a
    downstream component whose lexicon predates atomic chillu, and for
    round-trip tests.
    """
    from mlvoice.text.chars import CHILLU_TO_BASE

    return "".join(CHILLU_TO_BASE[ch] + VIRAMA + ZWJ if ch in CHILLU_TO_BASE else ch for ch in text)
