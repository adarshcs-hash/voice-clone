"""Malayalam numeral expansion.

Reading numbers aloud is where most Indic TTS pipelines visibly fail, because
Malayalam numerals are not a positional composition of digit names:

*   Tens combine with units through an oblique stem plus a ``യ`` glide before a
    vowel-initial unit: ``ഇരുപത്`` (20) + ``അഞ്ച്`` (5) -> ``ഇരുപത്തിയഞ്ച്``.
*   Hundreds are lexicalised: 500 is ``അഞ്ഞൂറ്``, not "five hundred".
*   Thousand multipliers are irregular: 3000 is ``മൂവായിരം``, 5000 is
    ``അയ്യായിരം``, 8000 is ``എണ്ണായിരം``, 10000 is ``പതിനായിരം``.
*   Grouping is the Indian lakh/crore system, not thousands-of-thousands.
*   The multiplier "one" is attributive ``ഒരു`` before a magnitude word, never
    ``ഒന്ന്``.

Scope and known simplification
------------------------------
Gemination sandhi across a magnitude boundary is deliberately **not** applied:
999 is rendered ``തൊള്ളായിരത്തിതൊണ്ണൂറ്റിയൊൻപത്`` where careful orthography
would geminate to ``തൊള്ളായിരത്തിത്തൊണ്ണൂറ്റിയൊൻപത്``. The two are homophonous
to a listener once :mod:`mlvoice.text.g2p` applies its own gemination rules, and
avoiding the sandhi keeps the tables auditable.

The tables in this module are the project's linguistic ground truth and are
covered exhaustively for 0-100 and by golden cases above that. Any change here
must be reviewed by a native speaker; see ``docs/malayalam-linguistics.md``.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Final

from mlvoice.errors import ValidationError

__all__ = [
    "MAX_CARDINAL",
    "cardinal",
    "decimal_number",
    "digit_by_digit",
    "multiplier",
    "ordinal",
]

CRORE: Final = 10_000_000
LAKH: Final = 100_000
MAX_CARDINAL: Final = 10**19 - 1
"""Upper bound. Above this the crore multiplier itself exceeds ten crore-crore."""

ZERO: Final = "പൂജ്യം"
MINUS: Final = "മൈനസ്"
POINT: Final = "പോയിന്റ്"
LAKH_WORD: Final = "ലക്ഷം"
CRORE_WORD: Final = "കോടി"
ONE_ATTRIBUTIVE: Final = "ഒരു"

_UNITS: Final[dict[int, str]] = {
    1: "ഒന്ന്",
    2: "രണ്ട്",
    3: "മൂന്ന്",
    4: "നാല്",
    5: "അഞ്ച്",
    6: "ആറ്",
    7: "ഏഴ്",
    8: "എട്ട്",
    9: "ഒൻപത്",
}

_TEENS: Final[dict[int, str]] = {
    10: "പത്ത്",
    11: "പതിനൊന്ന്",
    12: "പന്ത്രണ്ട്",
    13: "പതിമൂന്ന്",
    14: "പതിനാല്",
    15: "പതിനഞ്ച്",
    16: "പതിനാറ്",
    17: "പതിനേഴ്",
    18: "പതിനെട്ട്",
    19: "പത്തൊൻപത്",
}

_TENS: Final[dict[int, str]] = {
    20: "ഇരുപത്",
    30: "മുപ്പത്",
    40: "നാൽപത്",
    50: "അമ്പത്",
    60: "അറുപത്",
    70: "എഴുപത്",
    80: "എൺപത്",
    90: "തൊണ്ണൂറ്",
}

_TENS_STEM: Final[dict[int, str]] = {
    20: "ഇരുപത്തി",
    30: "മുപ്പത്തി",
    40: "നാൽപത്തി",
    50: "അമ്പത്തി",
    60: "അറുപത്തി",
    70: "എഴുപത്തി",
    80: "എൺപത്തി",
    90: "തൊണ്ണൂറ്റി",
}

_HUNDREDS: Final[dict[int, str]] = {
    1: "നൂറ്",
    2: "ഇരുനൂറ്",
    3: "മുന്നൂറ്",
    4: "നാനൂറ്",
    5: "അഞ്ഞൂറ്",
    6: "അറുനൂറ്",
    7: "എഴുനൂറ്",
    8: "എണ്ണൂറ്",
    9: "തൊള്ളായിരം",
}

_HUNDREDS_STEM: Final[dict[int, str]] = {
    1: "നൂറ്റി",
    2: "ഇരുനൂറ്റി",
    3: "മുന്നൂറ്റി",
    4: "നാനൂറ്റി",
    5: "അഞ്ഞൂറ്റി",
    6: "അറുനൂറ്റി",
    7: "എഴുനൂറ്റി",
    8: "എണ്ണൂറ്റി",
    9: "തൊള്ളായിരത്തി",
}

# Thousand multipliers 1-9 are irregular compounds, not unit + ആയിരം.
_UNIT_THOUSAND: Final[dict[int, str]] = {
    1: "ആയിരം",
    2: "രണ്ടായിരം",
    3: "മൂവായിരം",
    4: "നാലായിരം",
    5: "അയ്യായിരം",
    6: "ആറായിരം",
    7: "ഏഴായിരം",
    8: "എണ്ണായിരം",
    9: "ഒൻപതിനായിരം",
}

_TEEN_THOUSAND: Final[dict[int, str]] = {
    10: "പതിനായിരം",
    11: "പതിനൊന്നായിരം",
    12: "പന്ത്രണ്ടായിരം",
    13: "പതിമൂന്നായിരം",
    14: "പതിനാലായിരം",
    15: "പതിനയ്യായിരം",
    16: "പതിനാറായിരം",
    17: "പതിനേഴായിരം",
    18: "പതിനെട്ടായിരം",
    19: "പത്തൊൻപതിനായിരം",
}

_TENS_THOUSAND: Final[dict[int, str]] = {
    20: "ഇരുപതിനായിരം",
    30: "മുപ്പതിനായിരം",
    40: "നാൽപതിനായിരം",
    50: "അമ്പതിനായിരം",
    60: "അറുപതിനായിരം",
    70: "എഴുപതിനായിരം",
    80: "എൺപതിനായിരം",
    90: "തൊണ്ണൂറായിരം",
}

_DIGIT_NAMES: Final[dict[str, str]] = {"0": ZERO, **{str(k): v for k, v in _UNITS.items()}}

# Independent vowel letter -> the dependent sign used when it follows ``യ``.
_VOWEL_TO_SIGN: Final[dict[str, str]] = {
    "അ": "",
    "ആ": "ാ",
    "ഇ": "ി",
    "ഈ": "ീ",
    "ഉ": "ു",
    "ഊ": "ൂ",
    "ഋ": "ൃ",
    "എ": "െ",
    "ഏ": "േ",
    "ഐ": "ൈ",
    "ഒ": "ൊ",
    "ഓ": "ോ",
    "ഔ": "ൗ",
}


def _glide_join(stem: str, tail: str) -> str:
    """Join an oblique stem to a following word, inserting the ``യ`` glide.

    Malayalam does not allow the hiatus that a stem ending in ``ി`` followed by
    a vowel-initial word would create, so a ``യ`` is inserted and the initial
    vowel becomes its dependent sign::

        ഇരുപത്തി + അഞ്ച്  -> ഇരുപത്തിയഞ്ച്
        നൂറ്റി   + ഒന്ന്   -> നൂറ്റിയൊന്ന്
        ഇരുപത്തി + രണ്ട്   -> ഇരുപത്തിരണ്ട്   (consonant-initial: plain join)
    """
    if not tail:
        return stem
    first = tail[0]
    sign = _VOWEL_TO_SIGN.get(first)
    if sign is None:
        return stem + tail
    return stem + "യ" + sign + tail[1:]


def _stem(word: str) -> str:
    """Return the oblique stem of a magnitude word, used when a remainder follows.

    ``ആയിരം`` -> ``ആയിരത്തി``, ``ലക്ഷം`` -> ``ലക്ഷത്തി``, ``കോടി`` -> ``കോടി``.
    """
    if word.endswith("ം"):
        return word[:-1] + "ത്തി"
    if word.endswith("ി"):
        return word
    if word.endswith("റ്"):
        return word[:-2] + "റ്റി"
    if word.endswith("ത്"):
        return word[:-2] + "ത്തി"
    return word + "ി"


def _below_100(n: int) -> str:
    """Expand 1-99."""
    if n in _UNITS:
        return _UNITS[n]
    if n in _TEENS:
        return _TEENS[n]
    tens, unit = divmod(n, 10)
    if unit == 0:
        return _TENS[tens * 10]
    return _glide_join(_TENS_STEM[tens * 10], _UNITS[unit])


def _below_1000(n: int) -> str:
    """Expand 1-999."""
    if n < 100:
        return _below_100(n)
    hundreds, rest = divmod(n, 100)
    if rest == 0:
        return _HUNDREDS[hundreds]
    return _glide_join(_HUNDREDS_STEM[hundreds], _below_100(rest))


def _thousand_multiplier(k: int) -> str:
    """Expand ``k * 1000`` for 1 <= k <= 99, using the irregular compounds."""
    if k in _UNIT_THOUSAND:
        return _UNIT_THOUSAND[k]
    if k in _TEEN_THOUSAND:
        return _TEEN_THOUSAND[k]
    tens, unit = divmod(k, 10)
    if unit == 0:
        return _TENS_THOUSAND[tens * 10]
    return _glide_join(_TENS_STEM[tens * 10], _UNIT_THOUSAND[unit])


def _below_lakh(n: int) -> str:
    """Expand 1-99999."""
    if n < 1000:
        return _below_1000(n)
    thousands, rest = divmod(n, 1000)
    word = _thousand_multiplier(thousands)
    if rest == 0:
        return word
    return _glide_join(_stem(word), _below_1000(rest))


def multiplier(n: int) -> str:
    """Expand ``n`` as a multiplier of a magnitude word (``ലക്ഷം``, ``കോടി``).

    Malayalam uses the attributive ``ഒരു`` rather than ``ഒന്ന്`` in this slot:
    100000 is ``ഒരു ലക്ഷം``, never ``ഒന്ന് ലക്ഷം``.
    """
    return ONE_ATTRIBUTIVE if n == 1 else cardinal(n)


def cardinal(n: int) -> str:
    """Expand an integer to Malayalam words.

    Args:
        n: Integer in ``[-MAX_CARDINAL, MAX_CARDINAL]``.

    Returns:
        The number in Malayalam words, grouped by the lakh/crore system.

    Raises:
        ValidationError: ``n`` is outside the supported range.

    Examples:
        >>> cardinal(25)
        'ഇരുപത്തിയഞ്ച്'
        >>> cardinal(100000)
        'ഒരു ലക്ഷം'
    """
    if not isinstance(n, int):  # pragma: no cover - defensive, mypy enforces this
        raise ValidationError("cardinal() requires an int", got=type(n).__name__)
    if abs(n) > MAX_CARDINAL:
        raise ValidationError("number out of supported range", value=str(n))
    if n < 0:
        return f"{MINUS} {cardinal(-n)}"
    if n == 0:
        return ZERO

    parts: list[str] = []
    crores, n = divmod(n, CRORE)
    lakhs, rest = divmod(n, LAKH)

    if crores:
        word = CRORE_WORD if not (lakhs or rest) else _stem(CRORE_WORD)
        parts.append(f"{multiplier(crores)} {word}")
    if lakhs:
        word = LAKH_WORD if not rest else _stem(LAKH_WORD)
        parts.append(f"{multiplier(lakhs)} {word}")
    if rest:
        parts.append(_below_lakh(rest))
    return " ".join(parts)


def ordinal(n: int) -> str:
    """Expand an integer as an attributive ordinal (``ഒന്നാം``, ``രണ്ടാം``).

    Formed from the cardinal by dropping a final chandrakkala or anusvara and
    appending ``ാം``.
    """
    if n < 1:
        raise ValidationError("ordinal requires a positive integer", value=str(n))
    base = cardinal(n)
    if base.endswith("്") or base.endswith("ം"):
        base = base[:-1]
    return base + "ാം"


def digit_by_digit(digits: str, *, separator: str = " ") -> str:
    """Read a digit string one digit at a time.

    Used for phone numbers, PIN codes, OTPs and vehicle registrations, where a
    positional reading would be wrong.

    Raises:
        ValidationError: ``digits`` contains a non-digit character.
    """
    out: list[str] = []
    for ch in digits:
        name = _DIGIT_NAMES.get(ch)
        if name is None:
            raise ValidationError("digit_by_digit received a non-digit", char=ch)
        out.append(name)
    return separator.join(out)


def decimal_number(text: str) -> str:
    """Expand a decimal literal such as ``3.14`` or ``0.5``.

    The integer part is read positionally and the fractional digits one at a
    time, which is how decimals are read aloud in Malayalam.

    Raises:
        ValidationError: ``text`` is not a decimal literal.
    """
    try:
        Decimal(text)
    except InvalidOperation as exc:
        raise ValidationError("not a decimal literal", value=text) from exc

    negative = text.startswith("-")
    body = text.lstrip("+-")
    whole, _, frac = body.partition(".")
    whole_words = cardinal(int(whole)) if whole else ZERO
    if not frac:
        return f"{MINUS} {whole_words}" if negative else whole_words
    words = f"{whole_words} {POINT} {digit_by_digit(frac)}"
    return f"{MINUS} {words}" if negative else words
