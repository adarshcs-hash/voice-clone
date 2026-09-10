"""Manglish (romanised Malayalam) to Malayalam script transliteration.

Malayalam speakers routinely type Malayalam in Latin script -- "Manglish" --
and TTS input from chat, comments and search queries arrives that way. Reading
such input as English produces nonsense, so it must be converted to Malayalam
script before the phonemiser sees it.

Accuracy expectations
---------------------
Manglish is not a standardised orthography. The same Malayalam word has many
romanisations, and single Latin letters are genuinely ambiguous: ``t`` may be
``ത``, ``ട`` or ``റ്റ``; ``o`` may be ``ഒ`` or ``ഓ``. A rule system therefore
cannot be exact. This implementation is a deterministic, documented,
longest-match scheme that resolves the common cases plus a lexicon of overrides
for high-frequency words. Measure it with :mod:`mlvoice.eval` before relying on
it; the production answer for a general inbox is a trained sequence-to-sequence
transliterator, and :func:`transliterate` is the interface that model would slot
into.

The scheme follows the widely used Mozhi conventions where they are
unambiguous, and prefers the romanisation people actually type where they are
not (``ee`` -> ``ീ`` rather than ``േ``, ``oo`` -> ``ൂ``).
"""

from __future__ import annotations

import re
from typing import Final

from mlvoice.text.chars import BASE_TO_CHILLU, VIRAMA

__all__ = ["LEXICON", "MANGLISH_SIGNALS", "is_probably_manglish", "transliterate"]

# --- Tables -----------------------------------------------------------------

# Independent vowel letters, and the dependent sign used after a consonant.
_VOWELS: Final[dict[str, tuple[str, str]]] = {
    # romanisation: (independent, dependent sign)
    "a": ("അ", ""),
    "aa": ("ആ", "ാ"),
    "A": ("ആ", "ാ"),
    "i": ("ഇ", "ി"),
    "ee": ("ഈ", "ീ"),
    "ii": ("ഈ", "ീ"),
    "I": ("ഈ", "ീ"),
    "u": ("ഉ", "ു"),
    "oo": ("ഊ", "ൂ"),
    "uu": ("ഊ", "ൂ"),
    "U": ("ഊ", "ൂ"),
    "e": ("എ", "െ"),
    "ae": ("ഏ", "േ"),
    "E": ("ഏ", "േ"),
    "ai": ("ഐ", "ൈ"),
    "o": ("ഒ", "ൊ"),
    "O": ("ഓ", "ോ"),
    "oa": ("ഓ", "ോ"),
    "au": ("ഔ", "ൗ"),
    "ou": ("ഔ", "ൗ"),
    "ru": ("ഋ", "ൃ"),
}

_CONSONANTS: Final[dict[str, str]] = {
    # Velars
    "k": "ക",
    "kh": "ഖ",
    "g": "ഗ",
    "gh": "ഘ",
    "ng": "ങ",
    # Palatals
    "ch": "ച",
    "chh": "ഛ",
    "j": "ജ",
    "jh": "ഝ",
    "nj": "ഞ",
    # Retroflex
    "T": "ട",
    "Th": "ഠ",
    "D": "ഡ",
    "Dh": "ഢ",
    "N": "ണ",
    # Dentals
    "th": "ത",
    "thh": "ഥ",
    "d": "ദ",
    "dh": "ധ",
    "n": "ന",
    # Labials
    "p": "പ",
    "ph": "ഫ",
    "f": "ഫ",
    "b": "ബ",
    "bh": "ഭ",
    "m": "മ",
    # Approximants and liquids
    "y": "യ",
    "r": "ര",
    "R": "റ",
    "zh": "ഴ",
    "l": "ല",
    "L": "ള",
    "v": "വ",
    "w": "വ",
    # Sibilants and h
    "sh": "ഷ",
    "S": "ശ",
    "s": "സ",
    "h": "ഹ",
    # Frequent ambiguous singletons, resolved to the most common intent.
    "t": "ട",
    "z": "സ",
    "x": "ക്സ",
    "q": "ക",
    "c": "ക",
}

_ANUSVARA: Final = "ം"

# Words whose rule-based output is wrong often enough to be worth pinning.
# Kept small and auditable on purpose: it is a correction list, not a dictionary.
LEXICON: Final[dict[str, str]] = {
    "njan": "ഞാൻ",
    "ningal": "നിങ്ങൾ",
    "avan": "അവൻ",
    "aval": "അവൾ",
    "ente": "എന്റെ",
    "ende": "എന്റെ",
    "veedu": "വീട്",
    "vannu": "വന്നു",
    "poyi": "പോയി",
    "undu": "ഉണ്ട്",
    "illa": "ഇല്ല",
    "alla": "അല്ല",
    # The copula and negation family. Very high frequency, and the rules get
    # them wrong in a way a listener notices immediately: the retroflex ṇ of
    # ``ആണ്`` comes out as a dental ``ന``, and the long ``ഏ`` of ``അല്ലേ``
    # as a short ``എ``.
    "aanu": "ആണ്",
    "anu": "ആണ്",
    "aano": "ആണോ",
    "aanennu": "ആണെന്ന്",
    "alle": "അല്ലേ",
    "ille": "ഇല്ലേ",
    "venam": "വേണം",
    "venda": "വേണ്ട",
    "ariyilla": "അറിയില്ല",
    "ariyam": "അറിയാം",
    "enthu": "എന്ത്",
    "ethu": "ഏത്",
    "nanni": "നന്ദി",
    "nandi": "നന്ദി",
    "sugamano": "സുഖമാണോ",
    "kerala": "കേരളം",
    "keralam": "കേരളം",
    "malayalam": "മലയാളം",
    "thiruvananthapuram": "തിരുവനന്തപുരം",
    "kozhikode": "കോഴിക്കോട്",
    "ernakulam": "എറണാകുളം",
    "thrissur": "തൃശ്ശൂർ",
    "kochi": "കൊച്ചി",
    "onam": "ഓണം",
    "vishu": "വിഷു",
    "chetta": "ചേട്ടാ",
    "chechi": "ചേച്ചി",
    "ammu": "അമ്മു",
    "amma": "അമ്മ",
    "achan": "അച്ഛൻ",
    "sheri": "ശരി",
    "shari": "ശരി",
    "pinne": "പിന്നെ",
    "ippo": "ഇപ്പോൾ",
    "ippol": "ഇപ്പോൾ",
    "enthaanu": "എന്താണ്",
    "enthanu": "എന്താണ്",
    "engane": "എങ്ങനെ",
    "eppo": "എപ്പോൾ",
    "evide": "എവിടെ",
    "aara": "ആരാ",
    "aaru": "ആരു",
}

# Longest-match ordering. Sorting once at import keeps the scanner simple.
_CONSONANT_KEYS: Final[tuple[str, ...]] = tuple(sorted(_CONSONANTS, key=len, reverse=True))
_VOWEL_KEYS: Final[tuple[str, ...]] = tuple(sorted(_VOWELS, key=len, reverse=True))

_LATIN_WORD: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z']*")

# Positive evidence that a Latin-script word is romanised Malayalam rather than
# English: consonant digraphs Malayalam has and English mostly does not, and
# Malayalam inflectional endings. Requiring evidence rather than assuming it is
# what keeps ordinary English nouns ("world", "meeting", "file") out of the
# transliterator, where they would become unreadable Malayalam.
MANGLISH_SIGNALS: Final[tuple[str, ...]] = (
    "zh",
    "nj",
    "ng",
    "kk",
    "tt",
    "pp",
    "cch",
    "chch",
    "nn",
    "mm",
    "ll",
    "yy",
    "vv",
    "ss",
)
_MANGLISH_ENDINGS: Final[tuple[str, ...]] = (
    "unnu",
    "unnathu",
    "aanu",
    "aayi",
    "ikku",
    "kku",
    "inte",
    "ude",
    "ilu",
    "inu",
    "athu",
    "eedu",
    "aar",
    "aan",
    "um",
    "il",
    "oru",
)
_ASCII_ONLY: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z']+$")

# Latin letter sequences that mark a word as English rather than Manglish.
# Manglish is essentially syllabic (CV), so these clusters are strong signals.
_ENGLISH_MARKERS: Final[tuple[str, ...]] = (
    "tion",
    "ing",
    "ough",
    "wh",
    "qu",
    "ck",
    "ology",
    "ment",
    "able",
    "sion",
    "ture",
    "ness",
    "ance",
    "ence",
)

# High-frequency English words that survive the pattern heuristics. Code-mixed
# Malayalam is full of these, and transliterating them produces gibberish.
# Deliberately short: it covers function words and the technology/business
# vocabulary that dominates real code-mixed input, not English at large.
_ENGLISH_WORDS: Final[frozenset[str]] = frozenset(
    """
    a an the and or but if then than so because of for to from with without
    in on at by as is am are was were be been being do does did done have has
    had will would shall should can could may might must not no yes very more
    most less least all any some each every other another such same own
    i you he she it we they me him her us them my your his its our their this
    that these those what which who whom whose when where why how
    work works working home office school college student teacher company
    team meeting project report update message email phone mobile number
    data model server client user account login password api link file folder
    video audio image photo camera screen page site web app software hardware
    market price offer order payment bank card cash credit debit loan tax
    doctor hospital medicine health fitness food water train bus flight ticket
    time date today tomorrow yesterday morning evening night week month year
    good bad best better great nice new old free full open close start stop
    please thanks thank sorry welcome hello hey ok okay right left top down
    news live share like comment post follow channel group admin
    world method system service support product feature version release
    """.split()
)


class _Token:
    __slots__ = ("kind", "value")

    def __init__(self, kind: str, value: str) -> None:
        self.kind = kind  # "C", "V" or "X"
        self.value = value


def _scan(word: str) -> list[_Token]:
    """Segment a romanised word into consonant and vowel tokens, longest match first."""
    tokens: list[_Token] = []
    i = 0
    n = len(word)
    while i < n:
        for key in _CONSONANT_KEYS:
            if word.startswith(key, i):
                tokens.append(_Token("C", _CONSONANTS[key]))
                i += len(key)
                break
        else:
            for key in _VOWEL_KEYS:
                if word.startswith(key, i):
                    tokens.append(_Token("V", key))
                    i += len(key)
                    break
            else:
                tokens.append(_Token("X", word[i]))
                i += 1
    return tokens


def _assemble(tokens: list[_Token]) -> str:
    """Build Malayalam orthography from the token stream.

    A consonant takes the following vowel as a dependent sign; a consonant with
    no following vowel takes a chandrakkala, or the chillu form when it is
    word-final and a chillu exists.
    """
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        if token.kind == "X":
            out.append(token.value)
            i += 1
            continue
        if token.kind == "V":
            out.append(_VOWELS[token.value][0])
            i += 1
            continue
        # Consonant.
        letter = token.value
        nxt = tokens[i + 1] if i + 1 < n else None
        if nxt is not None and nxt.kind == "V":
            out.append(letter + _VOWELS[nxt.value][1])
            i += 2
            continue
        # No vowel follows: either a cluster or word-final.
        is_final = nxt is None
        if is_final and letter in BASE_TO_CHILLU:
            out.append(BASE_TO_CHILLU[letter])
        elif is_final and letter == "മ":
            out.append(_ANUSVARA)
        else:
            out.append(letter + VIRAMA)
        i += 1
    return "".join(out)


def is_probably_manglish(word: str, *, require_evidence: bool = True) -> bool:
    """Heuristic: does this Latin-script word look like romanised Malayalam?

    A word in :data:`LEXICON` always counts as Manglish.

    Rejected outright: high-frequency English words, words carrying orthographic
    patterns Malayalam syllable structure does not produce (``-tion``, ``-ing``,
    ``qu``, ``ck``), words with no vowel at all, and short all-caps tokens
    (acronyms such as ``BJP``, which are spelled out rather than
    transliterated).

    Args:
        word: A Latin-script token.
        require_evidence: When set (the default), the word must additionally
            carry a Malayalam signal -- one of :data:`MANGLISH_SIGNALS` or a
            Malayalam inflectional ending. This is the safer default for a
            general input stream: an English noun with no signal is left alone
            rather than transliterated into gibberish. Callers who know their
            input is Manglish (a chat or comment pipeline) can turn it off, at
            the cost of mangling any English in the same text.

    Examples:
        >>> is_probably_manglish("paranju"), is_probably_manglish("world")
        (True, False)
    """
    if not _ASCII_ONLY.match(word):
        return False
    lowered = word.lower()
    if lowered in _ENGLISH_WORDS:
        return False
    if any(marker in lowered for marker in _ENGLISH_MARKERS):
        return False
    if not any(ch in "aeiou" for ch in lowered):
        return False
    if word.isupper() and len(word) <= 5:
        return False
    if not require_evidence:
        return True
    # Being in the correction lexicon is itself evidence.
    if lowered in LEXICON:
        return True
    if any(signal in lowered for signal in MANGLISH_SIGNALS):
        return True
    return any(lowered.endswith(ending) for ending in _MANGLISH_ENDINGS)


def transliterate(text: str, *, use_lexicon: bool = True, require_evidence: bool = True) -> str:
    """Transliterate romanised Malayalam in ``text`` to Malayalam script.

    Only Latin-script runs are touched; Malayalam, digits and punctuation pass
    through unchanged. Words that :func:`is_probably_manglish` rejects are left
    as they are, so English embedded in Manglish survives.

    Args:
        text: Input that may contain Latin-script Malayalam.
        use_lexicon: Consult :data:`LEXICON` before applying rules.
        require_evidence: Passed to :func:`is_probably_manglish`.

    Returns:
        ``text`` with Manglish runs rewritten in Malayalam script.

    Examples:
        >>> transliterate("njan veedu poyi")
        'ഞാൻ വീട് പോയി'
    """

    def _one(match: re.Match[str]) -> str:
        word = match.group(0)
        if use_lexicon:
            pinned = LEXICON.get(word.lower())
            if pinned is not None:
                return pinned
        if not is_probably_manglish(word, require_evidence=require_evidence):
            return word
        # A capitalised proper noun carries an orthographic capital, not the
        # Mozhi retroflex/long marker, so scan it lowercased.
        scanned = word.lower() if word[:1].isupper() and word[1:].islower() else word
        return _assemble(_scan(scanned))

    return _LATIN_WORD.sub(_one, text)
