"""Code-mix segmentation and routing.

Real Malayalam input is rarely monolingual. A single sentence routinely mixes
Malayalam script, English words in Latin script, romanised Malayalam, acronyms
and digits::

    "Ente work laptop-il ready aanu, PDF file 5 MB und."

Each of those needs different treatment, so the pipeline first segments the
input into maximal runs of one script and then routes each run:

============  ==========================================================
Run kind      Treatment
============  ==========================================================
Malayalam     Passed through; already handled by normalisation and G2P.
Manglish      Transliterated to Malayalam script.
Acronym       Spelled out with Malayalam letter names (``BJP`` -> ബി ജെ പി).
English       Governed by :class:`LatinPolicy`; a loanword lexicon wins.
Digit/other   Already resolved by :mod:`mlvoice.text.expand`.
============  ==========================================================

On English words specifically
----------------------------
There is no honest rule-based way to turn arbitrary English orthography into
Malayalam script at production quality -- English spelling is not phonemic, so
the mapping needs a pronunciation model. This module therefore does not pretend
to: the default policy passes English through unchanged, an injectable loanword
lexicon pins the vocabulary that matters for a given deployment, and
:class:`LatinPolicy.TRANSLITERATE` is the seam where a measured
grapheme-to-phoneme model for English is dropped in. See
``docs/roadmap.md``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from mlvoice.text.chars import is_malayalam
from mlvoice.text.translit import is_probably_manglish, transliterate

__all__ = [
    "LATIN_LETTER_NAMES",
    "CodeMixConfig",
    "LatinPolicy",
    "ManglishDetection",
    "Script",
    "ScriptRun",
    "is_acronym",
    "route",
    "segment",
    "spell_acronym",
]

# Malayalam names for the Latin letters, used when spelling out an acronym.
LATIN_LETTER_NAMES: Final[dict[str, str]] = {
    "a": "ഏ",
    "b": "ബി",
    "c": "സി",
    "d": "ഡി",
    "e": "ഇ",
    "f": "എഫ്",
    "g": "ജി",
    "h": "എച്ച്",
    "i": "ഐ",
    "j": "ജെ",
    "k": "കെ",
    "l": "എൽ",
    "m": "എം",
    "n": "എൻ",
    "o": "ഒ",
    "p": "പി",
    "q": "ക്യൂ",
    "r": "ആർ",
    "s": "എസ്",
    "t": "ടി",
    "u": "യു",
    "v": "വി",
    "w": "ഡബ്ല്യു",
    "x": "എക്സ്",
    "y": "വൈ",
    "z": "സെഡ്",
}

_ACRONYM: Final[re.Pattern[str]] = re.compile(r"^(?:[A-Z]\.?){2,6}$")
_LATIN_RUN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z'.-]*")


class Script(StrEnum):
    """Script class of a text run."""

    MALAYALAM = "malayalam"
    LATIN = "latin"
    DIGIT = "digit"
    OTHER = "other"


class ManglishDetection(StrEnum):
    """How eagerly a Latin-script word is treated as romanised Malayalam."""

    EVIDENCE = "evidence"
    """Require a Malayalam signal (digraph or inflectional ending). The default:
    an English noun with no signal is left alone rather than transliterated."""
    PERMISSIVE = "permissive"
    """Transliterate anything that is not obviously English. For sources known
    to be Manglish, such as a chat or comment pipeline."""
    OFF = "off"
    """Never transliterate; only the lexicon applies."""


class LatinPolicy(StrEnum):
    """What to do with a Latin-script run that is not Manglish or an acronym."""

    KEEP = "keep"
    """Pass through unchanged. The default: honest about what we cannot do well."""
    TRANSLITERATE = "transliterate"
    """Apply the Manglish rules anyway. Use only for known-Manglish sources."""
    SPELL_OUT = "spell_out"
    """Spell the word letter by letter. Safe but slow to listen to."""
    DROP = "drop"
    """Remove the run. For pipelines that must never emit non-Malayalam audio."""


@dataclass(frozen=True, slots=True)
class ScriptRun:
    """A maximal run of one script within the input."""

    script: Script
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class CodeMixConfig:
    """Routing behaviour.

    Attributes:
        latin_policy: Treatment for English-looking Latin runs.
        manglish_detection: How eagerly to treat a Latin word as Manglish.
        spell_acronyms: Expand all-caps acronyms with Malayalam letter names.
        loanword_lexicon: Explicit Latin -> Malayalam mappings, consulted first
            and matched case-insensitively. Populate this per deployment with
            the domain vocabulary that must be pronounced correctly.
    """

    latin_policy: LatinPolicy = LatinPolicy.KEEP
    manglish_detection: ManglishDetection = ManglishDetection.EVIDENCE
    spell_acronyms: bool = True
    loanword_lexicon: Mapping[str, str] = field(default_factory=dict)
    folded_lexicon: Mapping[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "folded_lexicon",
            {key.lower(): value for key, value in self.loanword_lexicon.items()},
        )

    def lookup(self, word: str) -> str | None:
        """Resolve ``word`` through the lexicon, exact match first."""
        exact = self.loanword_lexicon.get(word)
        if exact is not None:
            return exact
        return self.folded_lexicon.get(word.lower())


DEFAULT_CONFIG: Final = CodeMixConfig()


def _classify(ch: str) -> Script:
    if is_malayalam(ch):
        return Script.MALAYALAM
    if ch.isascii() and ch.isalpha():
        return Script.LATIN
    if ch.isdigit():
        return Script.DIGIT
    return Script.OTHER


def segment(text: str) -> list[ScriptRun]:
    """Split ``text`` into maximal runs of a single script.

    Whitespace and punctuation form :attr:`Script.OTHER` runs, which keeps run
    boundaries aligned with word boundaries and makes the result losslessly
    re-joinable.

    Examples:
        >>> [(r.script.value, r.text) for r in segment("ഞാൻ work")]
        [('malayalam', 'ഞാൻ'), ('other', ' '), ('latin', 'work')]
    """
    if not text:
        return []
    runs: list[ScriptRun] = []
    start = 0
    current = _classify(text[0])
    for index, ch in enumerate(text[1:], start=1):
        kind = _classify(ch)
        if kind is not current:
            runs.append(ScriptRun(current, text[start:index], start, index))
            start, current = index, kind
    runs.append(ScriptRun(current, text[start:], start, len(text)))
    return runs


def is_acronym(word: str) -> bool:
    """True if ``word`` looks like an initialism that should be spelled out.

    Requires 2-6 letters, all upper case, optionally dot-separated. Mixed case
    is excluded so that a capitalised proper noun is not spelled out.

    Examples:
        >>> is_acronym("BJP"), is_acronym("U.P.S.C"), is_acronym("Kerala")
        (True, True, False)
    """
    if not _ACRONYM.match(word):
        return False
    letters = word.replace(".", "")
    return letters.isupper() and 2 <= len(letters) <= 6


def spell_acronym(word: str) -> str:
    """Spell an initialism using Malayalam letter names.

    Examples:
        >>> spell_acronym("BJP")
        'ബി ജെ പി'
    """
    letters = [ch for ch in word.lower() if ch.isalpha()]
    return " ".join(LATIN_LETTER_NAMES.get(ch, ch) for ch in letters)


def _route_latin_word(word: str, config: CodeMixConfig) -> str:
    pinned = config.lookup(word)
    if pinned is not None:
        return pinned
    if config.spell_acronyms and is_acronym(word):
        return spell_acronym(word)
    if config.manglish_detection is not ManglishDetection.OFF:
        require_evidence = config.manglish_detection is ManglishDetection.EVIDENCE
        if is_probably_manglish(word, require_evidence=require_evidence):
            return transliterate(word, require_evidence=require_evidence)
    match config.latin_policy:
        case LatinPolicy.KEEP:
            return word
        case LatinPolicy.TRANSLITERATE:
            return transliterate(word, use_lexicon=True, require_evidence=False)
        case LatinPolicy.SPELL_OUT:
            return spell_acronym(word)
        case LatinPolicy.DROP:
            return ""


def route(text: str, config: CodeMixConfig = DEFAULT_CONFIG) -> str:
    """Rewrite every Latin-script word in ``text`` according to ``config``.

    Malayalam, digits and punctuation are untouched, so this is safe to apply
    after :func:`mlvoice.text.expand.expand`.

    Examples:
        >>> route("njan BJP work cheythu")
        'ഞാൻ ബി ജെ പി work ചെയ്തു'
    """
    routed = _LATIN_RUN.sub(lambda m: _route_latin_word(m.group(0), config), text)
    # DROP and lexicon substitutions can leave doubled spaces behind.
    return re.sub(r"[^\S\n]{2,}", " ", routed).strip()
