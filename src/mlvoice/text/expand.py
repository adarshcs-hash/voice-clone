"""Text-level expansion of non-lexical tokens into Malayalam words.

This is the "text normalisation" stage in classical TTS terms: everything that
is written in a notation rather than in words -- digits, currency, percentages,
dates, clock times, units and abbreviations -- becomes the words a reader would
actually say.

Order is load-bearing. Dates and clock times must be recognised before bare
integers, or ``10/09/2026`` degenerates into three unrelated numbers. Each rule
is therefore a regex applied in a fixed sequence, and the sequence is covered by
tests rather than left to reviewer memory.

Digit strings long enough to be an identifier (phone number, PIN, OTP, vehicle
registration) are read digit by digit, because a positional reading of a phone
number is always wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from mlvoice.text.numbers import cardinal, decimal_number, digit_by_digit, ordinal

__all__ = ["ExpansionConfig", "expand"]

# --- Lexicons ---------------------------------------------------------------

MONTHS: Final[dict[int, str]] = {
    1: "ജനുവരി",
    2: "ഫെബ്രുവരി",
    3: "മാർച്ച്",
    4: "ഏപ്രിൽ",
    5: "മെയ്",
    6: "ജൂൺ",
    7: "ജൂലൈ",
    8: "ഓഗസ്റ്റ്",
    9: "സെപ്റ്റംബർ",
    10: "ഒക്ടോബർ",
    11: "നവംബർ",
    12: "ഡിസംബർ",
}

CURRENCIES: Final[dict[str, str]] = {
    "₹": "രൂപ",
    "$": "ഡോളർ",
    "€": "യൂറോ",
    "£": "പൗണ്ട്",
    "¥": "യെൻ",
}

# Written abbreviation -> spoken form. Keys are matched longest-first so that
# ``കി.മീ.`` wins over ``മീ.``.
ABBREVIATIONS: Final[dict[str, str]] = {
    # Malayalam titles and honorifics
    "ഡോ.": "ഡോക്ടർ",
    "പ്രൊഫ.": "പ്രൊഫസർ",
    "അഡ്വ.": "അഡ്വക്കേറ്റ്",
    "എൻജി.": "എൻജിനീയർ",
    # Malayalam units
    "കി.മീ.": "കിലോമീറ്റർ",
    "കി.മീ": "കിലോമീറ്റർ",
    "കി.ഗ്രാം": "കിലോഗ്രാം",
    "സെ.മീ.": "സെന്റിമീറ്റർ",
    "സെ.മീ": "സെന്റിമീറ്റർ",
    "ച.അടി": "ചതുരശ്ര അടി",
    "ശ.മാ.": "ശതമാനം",
    "രൂ.": "രൂപ",
    "നം.": "നമ്പർ",
    # Latin abbreviations that appear verbatim in Malayalam copy
    "Dr.": "ഡോക്ടർ",
    "Mr.": "മിസ്റ്റർ",
    "Mrs.": "മിസിസ്",
    "Ms.": "മിസ്",
    "Prof.": "പ്രൊഫസർ",
    "Rs.": "രൂപ",
    "No.": "നമ്പർ",
    "vs.": "വേഴ്സസ്",
    "etc.": "മുതലായവ",
}

UNITS: Final[dict[str, str]] = {
    "km": "കിലോമീറ്റർ",
    "kg": "കിലോഗ്രാം",
    "cm": "സെന്റിമീറ്റർ",
    "mm": "മില്ലിമീറ്റർ",
    "ml": "മില്ലിലിറ്റർ",
    "gb": "ജിഗാബൈറ്റ്",
    "mb": "മെഗാബൈറ്റ്",
    "kb": "കിലോബൈറ്റ്",
    "hz": "ഹെർട്സ്",
    "kw": "കിലോവാട്ട്",
    "m": "മീറ്റർ",
    "g": "ഗ്രാം",
    "l": "ലിറ്റർ",
}

PERCENT: Final = "ശതമാനം"
DEGREE_CELSIUS: Final = "ഡിഗ്രി സെൽഷ്യസ്"
HOUR: Final = "മണി"
MINUTE: Final = "മിനിറ്റ്"
SECOND: Final = "സെക്കൻഡ്"
PLUS_SPOKEN: Final = "പ്ലസ്"


@dataclass(frozen=True, slots=True)
class ExpansionConfig:
    """Tuning for :func:`expand`.

    Attributes:
        identifier_digit_threshold: A separator-free digit run at least this
            long is read digit by digit rather than positionally. Ten is the
            length of an Indian mobile number; six catches PIN codes and OTPs.
        expand_abbreviations: Apply the :data:`ABBREVIATIONS` lexicon.
        expand_units: Apply the :data:`UNITS` lexicon after a number.
        day_first: Interpret an ambiguous ``a/b/c`` date as day/month/year
            (Indian convention) rather than month/day/year.
    """

    identifier_digit_threshold: int = 6
    expand_abbreviations: bool = True
    expand_units: bool = True
    day_first: bool = True


DEFAULT_CONFIG: Final = ExpansionConfig()

# --- Patterns ---------------------------------------------------------------

_NUM = r"\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?"

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_SLASH_DATE = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b")
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm|AM|PM)?\b")
_CURRENCY = re.compile(rf"([{''.join(re.escape(c) for c in CURRENCIES)}])\s?({_NUM})")
_PERCENT = re.compile(rf"({_NUM})\s?%")
_CELSIUS = re.compile(rf"({_NUM})\s?°\s?C\b")
_UNIT_ALTERNATION = "|".join(sorted(UNITS, key=len, reverse=True))
_UNIT = re.compile(rf"({_NUM})\s?({_UNIT_ALTERNATION})\b", re.IGNORECASE)
_ENGLISH_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_MALAYALAM_ORDINAL = re.compile(r"\b(\d+)ാം")
_PHONE_INTL = re.compile(r"\+(\d{1,3})[\s-]?(\d{6,14})\b")
_DIGIT_GROUPED = re.compile(r"\b\d{1,3}(?:,\d{2,3})+(?:\.\d+)?\b")
_DECIMAL = re.compile(r"\b\d+\.\d+\b")
_INTEGER = re.compile(r"\d+")


def _spoken_time(hour: int, minute: int, second: int | None, meridiem: str | None) -> str:
    parts = [f"{cardinal(hour)} {HOUR}"]
    if minute:
        parts.append(f"{cardinal(minute)} {MINUTE}")
    if second:
        parts.append(f"{cardinal(second)} {SECOND}")
    if meridiem:
        # രാവിലെ / വൈകുന്നേരം read more naturally than a transliterated am/pm.
        parts.insert(0, "രാവിലെ" if meridiem.lower() == "am" else "വൈകുന്നേരം")
    return " ".join(parts)


def _spoken_date(year: int, month: int, day: int) -> str:
    month_name = MONTHS.get(month)
    if month_name is None or not 1 <= day <= 31:
        # Not a real date after all; fall back to reading the components.
        return f"{cardinal(year)} {cardinal(month)} {cardinal(day)}"
    return f"{cardinal(year)} {month_name} {cardinal(day)}"


def _strip_group_separators(literal: str) -> str:
    return literal.replace(",", "")


def _spoken_quantity(literal: str) -> str:
    """Expand a numeric literal that may carry Indian digit grouping."""
    plain = _strip_group_separators(literal)
    if "." in plain:
        return decimal_number(plain)
    return cardinal(int(plain))


def _expand_abbreviations(text: str) -> str:
    for written in sorted(ABBREVIATIONS, key=len, reverse=True):
        if written in text:
            text = text.replace(written, ABBREVIATIONS[written])
    return text


def expand(text: str, config: ExpansionConfig = DEFAULT_CONFIG) -> str:
    """Expand every non-lexical token in ``text`` into Malayalam words.

    Args:
        text: Unicode-normalised text (run :func:`mlvoice.text.unicode_norm.normalize`
            first; Malayalam digits must already be ASCII).
        config: Behaviour tuning.

    Returns:
        Text in which numbers, dates, times, currency, units, percentages and
        known abbreviations have been replaced by their spoken Malayalam forms.

    Examples:
        >>> expand("₹250 വേണം")
        'ഇരുനൂറ്റിയമ്പത് രൂപ വേണം'
    """
    if config.expand_abbreviations:
        text = _expand_abbreviations(text)

    # 1. International phone numbers, before anything strips the ``+``.
    text = _PHONE_INTL.sub(
        lambda m: f"{PLUS_SPOKEN} {digit_by_digit(m.group(1))} {digit_by_digit(m.group(2))}",
        text,
    )

    # 2. Dates, before bare integers can consume the components.
    text = _ISO_DATE.sub(
        lambda m: _spoken_date(int(m.group(1)), int(m.group(2)), int(m.group(3))), text
    )

    def _slash(m: re.Match[str]) -> str:
        a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        day, month = (a, b) if config.day_first else (b, a)
        return _spoken_date(year, month, day)

    text = _SLASH_DATE.sub(_slash, text)

    # 3. Clock times.
    text = _TIME.sub(
        lambda m: _spoken_time(
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)) if m.group(3) else None,
            m.group(4),
        ),
        text,
    )

    # 4. Ordinals, before the integer rule.
    text = _MALAYALAM_ORDINAL.sub(lambda m: ordinal(int(m.group(1))), text)
    text = _ENGLISH_ORDINAL.sub(lambda m: ordinal(int(m.group(1))), text)

    # 5. Currency, percentage, temperature, units.
    text = _CURRENCY.sub(lambda m: f"{_spoken_quantity(m.group(2))} {CURRENCIES[m.group(1)]}", text)
    text = _PERCENT.sub(lambda m: f"{_spoken_quantity(m.group(1))} {PERCENT}", text)
    text = _CELSIUS.sub(lambda m: f"{_spoken_quantity(m.group(1))} {DEGREE_CELSIUS}", text)
    if config.expand_units:
        text = _UNIT.sub(
            lambda m: f"{_spoken_quantity(m.group(1))} {UNITS[m.group(2).lower()]}", text
        )

    # 6. Grouped numbers (1,25,000), decimals, then bare integers.
    text = _DIGIT_GROUPED.sub(lambda m: _spoken_quantity(m.group(0)), text)
    text = _DECIMAL.sub(lambda m: decimal_number(m.group(0)), text)

    def _integer(m: re.Match[str]) -> str:
        literal = m.group(0)
        if len(literal) >= config.identifier_digit_threshold:
            return digit_by_digit(literal)
        return cardinal(int(literal))

    return _INTEGER.sub(_integer, text)
