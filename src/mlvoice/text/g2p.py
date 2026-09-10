"""Malayalam grapheme-to-phoneme conversion.

Malayalam orthography is close to phonemic but not identical to pronunciation.
Four systematic gaps account for most of the mispronunciation heard from
off-the-shelf multilingual TTS on Malayalam:

1.  **Samvruthokaram** -- the half-``u``. A word-final chandrakkala, and a
    word-final ``ു``, are both realised as a short central vowel [ɨ]:
    ``നാട്`` is [naːɖɨ] and ``അതു`` is [ad̪ɨ]. Rendering either as a bare
    consonant or as a full [u] is the single most audible error a Malayalam
    voice can make.
2.  **Gemination** -- a doubled consonant is phonemically long. ``പത്ത്`` is
    [pat̪ːɨ], distinct from a single [t̪].
3.  **Homorganic nasal assimilation and post-nasal voicing** -- the anusvara
    ``ം`` takes the place of articulation of a following stop, and a nasal
    followed by a voiceless stop voices it: ``ങ്ക`` is [ŋɡ], ``ഞ്ച`` is [ɲdʒ],
    ``ണ്ട`` is [ɳɖ], ``മ്പ`` is [mb].
4.  **Cluster idiosyncrasies** -- ``റ്റ`` is a geminate alveolar stop [tː], not
    a doubled trill, and ``ന്റ`` is [nd̪], not [nr].

Everything above is systematic and implemented here. Allophonic *lenition* of
intervocalic single stops (``ക`` -> [ɡ]/[ɣ]) is real but dialect- and
register-dependent, so it is available behind ``intervocalic_voicing`` and
**off by default**: a neural acoustic model trained on real speech learns that
variation on its own, and forcing one dialect's realisation into the phoneme
string removes the model's ability to.

Output
------
:func:`phonemize` returns phonemes as a list of tokens. Each token has a stable
ASCII identifier alongits IPA form (see :data:`PHONEME_TABLE`), because model
tokenizers handle combining diacritics inconsistently and a stable ASCII symbol
set makes vocabularies reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from mlvoice.text.chars import (
    ANUSVARA,
    CANDRABINDU,
    VIRAMA,
    VISARGA,
)
from mlvoice.text.chars import (
    CHILLU_TO_BASE as _CHILLU_TO_BASE,
)

__all__ = [
    "PHONEME_TABLE",
    "G2PConfig",
    "Notation",
    "phonemize",
    "phonemize_word",
    "to_string",
]

SAMVRUTHOKARAM: Final = "ɨ"
LENGTH_MARK: Final = "ː"
WORD_BOUNDARY: Final = "|"

# --- Base grapheme -> phoneme tables ----------------------------------------

_CONSONANTS: Final[dict[str, str]] = {
    "ക": "k",
    "ഖ": "kʰ",
    "ഗ": "ɡ",
    "ഘ": "ɡʰ",
    "ങ": "ŋ",
    "ച": "tʃ",
    "ഛ": "tʃʰ",
    "ജ": "dʒ",
    "ഝ": "dʒʰ",
    "ഞ": "ɲ",
    "ട": "ʈ",
    "ഠ": "ʈʰ",
    "ഡ": "ɖ",
    "ഢ": "ɖʰ",
    "ണ": "ɳ",
    "ത": "t̪",
    "ഥ": "t̪ʰ",
    "ദ": "d̪",
    "ധ": "d̪ʰ",
    "ന": "n",
    "ഩ": "n",
    "പ": "p",
    "ഫ": "pʰ",
    "ബ": "b",
    "ഭ": "bʰ",
    "മ": "m",
    "യ": "j",
    "ര": "ɾ",
    "റ": "r",
    "ല": "l",
    "ള": "ɭ",
    "ഴ": "ɻ",
    "വ": "ʋ",
    "ശ": "ʃ",
    "ഷ": "ʂ",
    "സ": "s",
    "ഹ": "h",
    "ഺ": "t",
}

_INDEPENDENT_VOWELS: Final[dict[str, str]] = {
    "അ": "a",
    "ആ": "aː",
    "ഇ": "i",
    "ഈ": "iː",
    "ഉ": "u",
    "ഊ": "uː",
    "ഋ": "ɾɨ",
    "ഌ": "lɨ",
    "എ": "e",
    "ഏ": "eː",
    "ഐ": "ai",
    "ഒ": "o",
    "ഓ": "oː",
    "ഔ": "au",
}

_VOWEL_SIGNS: Final[dict[str, str]] = {
    "ാ": "aː",
    "ി": "i",
    "ീ": "iː",
    "ു": "u",
    "ൂ": "uː",
    "ൃ": "ɾɨ",
    "ൄ": "ɾɨː",
    "െ": "e",
    "േ": "eː",
    "ൈ": "ai",
    "ൊ": "o",
    "ോ": "oː",
    "ൌ": "au",
    "ൗ": "au",
    "ൢ": "lɨ",
    "ൣ": "lɨː",
}

_CHILLU_PHONEMES: Final[dict[str, str]] = {
    chillu: _CONSONANTS[base] if base != "ര" else "r" for chillu, base in _CHILLU_TO_BASE.items()
}

# --- Phonological rule tables -----------------------------------------------

_NASALS: Final[frozenset[str]] = frozenset({"m", "n", "ɳ", "ɲ", "ŋ"})

# Place assimilation for the anusvara, keyed by the following consonant.
_HOMORGANIC_NASAL: Final[dict[str, str]] = {
    "k": "ŋ",
    "kʰ": "ŋ",
    "ɡ": "ŋ",
    "ɡʰ": "ŋ",
    "tʃ": "ɲ",
    "tʃʰ": "ɲ",
    "dʒ": "ɲ",
    "ʈ": "ɳ",
    "ʈʰ": "ɳ",
    "ɖ": "ɳ",
    "t̪": "n",
    "t̪ʰ": "n",
    "d̪": "n",
    "n": "n",
    "p": "m",
    "pʰ": "m",
    "b": "m",
    "m": "m",
}

# Voiceless stop -> voiced counterpart, for post-nasal voicing and lenition.
_VOICED: Final[dict[str, str]] = {
    "k": "ɡ",
    "tʃ": "dʒ",
    "ʈ": "ɖ",
    "t̪": "d̪",
    "p": "b",
}

# Clusters whose realisation is not the concatenation of their parts.
_SPECIAL_CLUSTERS: Final[dict[tuple[str, str], tuple[str, ...]]] = {
    ("r", "r"): ("tː",),  # റ്റ -> geminate alveolar stop
    ("n", "r"): ("n", "d̪"),  # ന്റ -> [nd̪]
}

# Stable ASCII identifier for every phoneme this module can emit. Model
# vocabularies are built from these, never from the IPA strings.
PHONEME_TABLE: Final[dict[str, str]] = {
    # vowels
    "a": "a",
    "aː": "aa",
    "i": "i",
    "iː": "ii",
    "u": "u",
    "uː": "uu",
    "ɨ": "ax",
    "e": "e",
    "eː": "ee",
    "ai": "ai",
    "o": "o",
    "oː": "oo",
    "au": "au",
    "ɾɨ": "rx",
    "ɾɨː": "rxx",
    "lɨ": "lx",
    "lɨː": "lxx",
    # stops
    "k": "k",
    "kʰ": "kh",
    "ɡ": "g",
    "ɡʰ": "gh",
    "tʃ": "c",
    "tʃʰ": "ch",
    "dʒ": "j",
    "dʒʰ": "jh",
    "ʈ": "tt",
    "ʈʰ": "tth",
    "ɖ": "dd",
    "ɖʰ": "ddh",
    "t̪": "t",
    "t̪ʰ": "th",
    "d̪": "d",
    "d̪ʰ": "dh",
    "t": "tr",
    "p": "p",
    "pʰ": "ph",
    "b": "b",
    "bʰ": "bh",
    # nasals
    "ŋ": "ng",
    "ɲ": "ny",
    "ɳ": "nn",
    "n": "n",
    "m": "m",
    # approximants, liquids, fricatives
    "j": "y",
    "ɾ": "r",
    "r": "rr",
    "l": "l",
    "ɭ": "ll",
    "ɻ": "zh",
    "ʋ": "v",
    "ʃ": "sh",
    "ʂ": "ss",
    "s": "s",
    "h": "h",
    WORD_BOUNDARY: "|",
}


class Notation(StrEnum):
    """Output symbol set for :func:`to_string`."""

    IPA = "ipa"
    ASCII = "ascii"


@dataclass(frozen=True, slots=True)
class G2PConfig:
    """Rule switches.

    Attributes:
        intervocalic_voicing: Voice a single intervocalic voiceless stop
            (``ക`` -> [ɡ]). Dialect- and register-dependent; off by default so
            that the acoustic model learns the variation from data.
        post_nasal_voicing: Voice a voiceless stop after a homorganic nasal
            (``ണ്ട`` -> [ɳɖ]). Systematic in standard Malayalam.
        samvruthokaram: Realise a word-final chandrakkala or ``ു`` as [ɨ].
        geminate_clusters: Collapse a doubled consonant into a single long one.
    """

    intervocalic_voicing: bool = False
    post_nasal_voicing: bool = True
    samvruthokaram: bool = True
    geminate_clusters: bool = True


DEFAULT_CONFIG: Final = G2PConfig()


class _Kind(StrEnum):
    CONSONANT = "C"
    VOWEL = "V"


@dataclass(slots=True)
class _Segment:
    kind: _Kind
    phone: str
    from_virama: bool = False
    """The consonant was written with an explicit chandrakkala."""
    from_u_sign: bool = False
    """The vowel came from the ``ു`` sign, which may be samvruthokaram."""
    from_anusvara: bool = False
    """The consonant came from ``ം`` and is subject to place assimilation."""


def _parse(word: str) -> list[_Segment]:
    """Convert a Malayalam word into a flat consonant/vowel segment list."""
    segments: list[_Segment] = []
    index = 0
    length = len(word)
    while index < length:
        ch = word[index]
        if ch in _CONSONANTS:
            phone = _CONSONANTS[ch]
            nxt = word[index + 1] if index + 1 < length else ""
            if nxt == VIRAMA:
                segments.append(_Segment(_Kind.CONSONANT, phone, from_virama=True))
                index += 2
            elif nxt in _VOWEL_SIGNS:
                segments.append(_Segment(_Kind.CONSONANT, phone))
                segments.append(_Segment(_Kind.VOWEL, _VOWEL_SIGNS[nxt], from_u_sign=nxt == "ു"))
                index += 2
            else:
                # No sign: the inherent /a/ surfaces.
                segments.append(_Segment(_Kind.CONSONANT, phone))
                segments.append(_Segment(_Kind.VOWEL, "a"))
                index += 1
        elif ch in _CHILLU_PHONEMES:
            segments.append(_Segment(_Kind.CONSONANT, _CHILLU_PHONEMES[ch]))
            index += 1
        elif ch in _INDEPENDENT_VOWELS:
            segments.append(_Segment(_Kind.VOWEL, _INDEPENDENT_VOWELS[ch]))
            index += 1
        elif ch == ANUSVARA:
            segments.append(_Segment(_Kind.CONSONANT, "m", from_anusvara=True))
            index += 1
        elif ch == VISARGA:
            segments.append(_Segment(_Kind.CONSONANT, "h"))
            index += 1
        elif ch in {CANDRABINDU, VIRAMA}:
            # A stray virama with no preceding consonant, or a candrabindu we
            # have no separate phoneme for: nasalisation is carried by the
            # neighbouring vowel in this inventory.
            index += 1
        else:
            # Not Malayalam (punctuation, a stray Latin letter). Callers route
            # those away before this point; skipping keeps the phonemiser total.
            index += 1
    return segments


def _assimilate_anusvara(segments: list[_Segment]) -> None:
    """Give ``ം`` the place of articulation of the following consonant."""
    for i, seg in enumerate(segments):
        if not seg.from_anusvara:
            continue
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        if nxt is not None and nxt.kind is _Kind.CONSONANT:
            seg.phone = _HOMORGANIC_NASAL.get(nxt.phone, seg.phone)


def _apply_clusters(segments: list[_Segment]) -> list[_Segment]:
    """Resolve special clusters and collapse geminates."""
    out: list[_Segment] = []
    i = 0
    n = len(segments)
    while i < n:
        seg = segments[i]
        nxt = segments[i + 1] if i + 1 < n else None
        if (
            seg.kind is _Kind.CONSONANT
            and nxt is not None
            and nxt.kind is _Kind.CONSONANT
            and seg.from_virama
        ):
            special = _SPECIAL_CLUSTERS.get((seg.phone, nxt.phone))
            if special is not None:
                for phone in special:
                    out.append(_Segment(_Kind.CONSONANT, phone, from_virama=True))
                out[-1].from_virama = nxt.from_virama
                i += 2
                continue
            if seg.phone == nxt.phone:
                out.append(
                    _Segment(
                        _Kind.CONSONANT,
                        seg.phone + LENGTH_MARK,
                        from_virama=nxt.from_virama,
                    )
                )
                i += 2
                continue
        out.append(seg)
        i += 1
    return out


def _post_nasal_voicing(segments: list[_Segment]) -> None:
    """Voice a voiceless stop that follows a homorganic nasal."""
    for i in range(1, len(segments)):
        prev, seg = segments[i - 1], segments[i]
        if (
            prev.kind is _Kind.CONSONANT
            and prev.phone in _NASALS
            and seg.kind is _Kind.CONSONANT
            and seg.phone in _VOICED
        ):
            seg.phone = _VOICED[seg.phone]


def _samvruthokaram(segments: list[_Segment]) -> None:
    """Realise a word-final chandrakkala or ``ു`` as the half-vowel [ɨ]."""
    if not segments:
        return
    last = segments[-1]
    if last.kind is _Kind.CONSONANT and last.from_virama:
        segments.append(_Segment(_Kind.VOWEL, SAMVRUTHOKARAM))
    elif last.kind is _Kind.VOWEL and last.from_u_sign:
        last.phone = SAMVRUTHOKARAM


def _intervocalic_voicing(segments: list[_Segment]) -> None:
    """Voice a single voiceless stop standing between two vowels."""
    for i in range(1, len(segments) - 1):
        prev, seg, nxt = segments[i - 1], segments[i], segments[i + 1]
        if (
            seg.kind is _Kind.CONSONANT
            and seg.phone in _VOICED
            and prev.kind is _Kind.VOWEL
            and nxt.kind is _Kind.VOWEL
        ):
            seg.phone = _VOICED[seg.phone]


def phonemize_word(word: str, config: G2PConfig = DEFAULT_CONFIG) -> list[str]:
    """Convert a single normalised Malayalam word to phoneme tokens.

    Args:
        word: One Malayalam word. Run :func:`mlvoice.text.unicode_norm.normalize`
            first; non-Malayalam characters are skipped.
        config: Rule switches.

    Returns:
        Phoneme tokens in IPA notation.

    Examples:
        >>> phonemize_word("നാട്")
        ['n', 'aː', 'ʈ', 'ɨ']
        >>> phonemize_word("പത്ത്")
        ['p', 'a', 't̪ː', 'ɨ']
        >>> phonemize_word("എന്റെ")
        ['e', 'n', 'd̪', 'e']
    """
    segments = _parse(word)
    if not segments:
        return []
    _assimilate_anusvara(segments)
    if config.geminate_clusters:
        segments = _apply_clusters(segments)
    if config.post_nasal_voicing:
        _post_nasal_voicing(segments)
    if config.samvruthokaram:
        _samvruthokaram(segments)
    if config.intervocalic_voicing:
        _intervocalic_voicing(segments)
    return [seg.phone for seg in segments]


def phonemize(text: str, config: G2PConfig = DEFAULT_CONFIG) -> list[list[str]]:
    """Phonemise whitespace-separated Malayalam text, one token list per word.

    Words that contain no Malayalam produce an empty list, preserving alignment
    with the input token sequence so that callers can zip the two.
    """
    return [phonemize_word(word, config) for word in text.split()]


def to_string(
    words: list[list[str]],
    *,
    notation: Notation = Notation.IPA,
    word_boundary: str = WORD_BOUNDARY,
) -> str:
    """Render :func:`phonemize` output as a single string.

    Args:
        words: Per-word phoneme token lists.
        notation: ``IPA`` keeps the phonetic symbols; ``ASCII`` maps each to its
            stable identifier from :data:`PHONEME_TABLE`, which is what model
            vocabularies should be built from.
        word_boundary: Token inserted between words.

    Raises:
        KeyError: A phoneme has no ASCII identifier, which means
            :data:`PHONEME_TABLE` and the rule tables have drifted apart. The
            test suite asserts they cannot.
    """
    rendered: list[str] = []
    for tokens in words:
        if not tokens:
            continue
        if notation is Notation.ASCII:
            rendered.append(" ".join(_ascii_symbol(t) for t in tokens))
        else:
            rendered.append(" ".join(tokens))
    return f" {word_boundary} ".join(rendered)


def _ascii_symbol(phone: str) -> str:
    """Map one phoneme to its ASCII identifier, preserving the length mark."""
    if phone.endswith(LENGTH_MARK) and phone not in PHONEME_TABLE:
        return PHONEME_TABLE[phone[: -len(LENGTH_MARK)]] + ":"
    return PHONEME_TABLE[phone]
