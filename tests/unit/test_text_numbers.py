"""Malayalam numeral expansion and text-level expansion.

The parametrised cases here are the project's linguistic ground truth for
numbers. Changing an expected value means changing a claim about Malayalam, and
should be reviewed as such.
"""

from __future__ import annotations

import pytest

from mlvoice.errors import ValidationError
from mlvoice.text.expand import ExpansionConfig, expand
from mlvoice.text.numbers import (
    MAX_CARDINAL,
    cardinal,
    decimal_number,
    digit_by_digit,
    multiplier,
    ordinal,
)


class TestCardinalsBelowHundred:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "പൂജ്യം"),
            (1, "ഒന്ന്"),
            (5, "അഞ്ച്"),
            (9, "ഒൻപത്"),
            (10, "പത്ത്"),
            (11, "പതിനൊന്ന്"),
            (12, "പന്ത്രണ്ട്"),
            (15, "പതിനഞ്ച്"),
            (19, "പത്തൊൻപത്"),
            (20, "ഇരുപത്"),
            (30, "മുപ്പത്"),
            (40, "നാൽപത്"),
            (50, "അമ്പത്"),
            (60, "അറുപത്"),
            (70, "എഴുപത്"),
            (80, "എൺപത്"),
            (90, "തൊണ്ണൂറ്"),
        ],
    )
    def test_lexical_forms(self, value: int, expected: str) -> None:
        assert cardinal(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (21, "ഇരുപത്തിയൊന്ന്"),
            (22, "ഇരുപത്തിരണ്ട്"),
            (24, "ഇരുപത്തിനാല്"),
            (25, "ഇരുപത്തിയഞ്ച്"),
            (26, "ഇരുപത്തിയാറ്"),
            (28, "ഇരുപത്തിയെട്ട്"),
            (29, "ഇരുപത്തിയൊൻപത്"),
            (99, "തൊണ്ണൂറ്റിയൊൻപത്"),
        ],
    )
    def test_glide_sandhi(self, value: int, expected: str) -> None:
        """A vowel-initial unit takes a ``യ`` glide; a consonant-initial one does not."""
        assert cardinal(value) == expected

    def test_every_value_below_hundred_expands(self) -> None:
        for value in range(100):
            words = cardinal(value)
            assert words
            assert " " not in words, f"{value} should be one word, got {words!r}"


class TestLargerCardinals:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (100, "നൂറ്"),
            (101, "നൂറ്റിയൊന്ന്"),
            (102, "നൂറ്റിരണ്ട്"),
            (125, "നൂറ്റിയിരുപത്തിയഞ്ച്"),
            (500, "അഞ്ഞൂറ്"),
            (900, "തൊള്ളായിരം"),
            (1000, "ആയിരം"),
            (1001, "ആയിരത്തിയൊന്ന്"),
            (2500, "രണ്ടായിരത്തിയഞ്ഞൂറ്"),
            (3000, "മൂവായിരം"),
            (5000, "അയ്യായിരം"),
            (8000, "എണ്ണായിരം"),
            (9000, "ഒൻപതിനായിരം"),
            (10_000, "പതിനായിരം"),
            (25_000, "ഇരുപത്തിയയ്യായിരം"),
            (100_000, "ഒരു ലക്ഷം"),
            (125_000, "ഒരു ലക്ഷത്തി ഇരുപത്തിയയ്യായിരം"),
            (1_000_000, "പത്ത് ലക്ഷം"),
            (10_000_000, "ഒരു കോടി"),
        ],
    )
    def test_irregular_and_grouped_forms(self, value: int, expected: str) -> None:
        assert cardinal(value) == expected

    def test_lakh_crore_grouping_not_thousands(self) -> None:
        """Indian grouping: 10^6 is ten lakh, not one million."""
        assert cardinal(1_000_000).startswith("പത്ത് ലക്ഷം")

    def test_multiplier_one_is_attributive(self) -> None:
        assert multiplier(1) == "ഒരു"
        assert multiplier(2) == cardinal(2)

    def test_negative(self) -> None:
        assert cardinal(-5) == "മൈനസ് അഞ്ച്"

    def test_out_of_range_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            cardinal(MAX_CARDINAL + 1)

    def test_wide_sweep_does_not_raise(self) -> None:
        for value in range(0, 200_000, 997):
            assert cardinal(value)


class TestOrdinalsAndDecimals:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(1, "ഒന്നാം"), (2, "രണ്ടാം"), (3, "മൂന്നാം"), (10, "പത്താം"), (25, "ഇരുപത്തിയഞ്ചാം")],
    )
    def test_ordinals(self, value: int, expected: str) -> None:
        assert ordinal(value) == expected

    def test_ordinal_rejects_zero(self) -> None:
        with pytest.raises(ValidationError):
            ordinal(0)

    def test_decimal_reads_fraction_digit_wise(self) -> None:
        assert decimal_number("3.14") == "മൂന്ന് പോയിന്റ് ഒന്ന് നാല്"

    def test_decimal_without_fraction(self) -> None:
        assert decimal_number("7") == "ഏഴ്"

    def test_negative_decimal(self) -> None:
        assert decimal_number("-0.5").startswith("മൈനസ്")

    def test_decimal_rejects_non_numeric(self) -> None:
        with pytest.raises(ValidationError):
            decimal_number("abc")

    def test_digit_by_digit(self) -> None:
        assert digit_by_digit("905") == "ഒൻപത് പൂജ്യം അഞ്ച്"

    def test_digit_by_digit_rejects_letters(self) -> None:
        with pytest.raises(ValidationError):
            digit_by_digit("12a")


class TestExpansion:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("₹250 തന്നു", "ഇരുനൂറ്റിയമ്പത് രൂപ തന്നു"),
            ("1,25,000 രൂപ", "ഒരു ലക്ഷത്തി ഇരുപത്തിയയ്യായിരം രൂപ"),
            ("50% വളർച്ച", "അമ്പത് ശതമാനം വളർച്ച"),
            ("5 km ദൂരം", "അഞ്ച് കിലോമീറ്റർ ദൂരം"),
            ("25ാം തീയതി", "ഇരുപത്തിയഞ്ചാം തീയതി"),
            ("ഡോ. രാജൻ", "ഡോക്ടർ രാജൻ"),
        ],
    )
    def test_expansions(self, source: str, expected: str) -> None:
        assert expand(source) == expected

    def test_iso_date(self) -> None:
        out = expand("2026-09-10 ന്")
        assert "സെപ്റ്റംബർ" in out
        assert "രണ്ടായിരത്തിയിരുപത്തിയാറ്" in out

    def test_day_first_is_the_default(self) -> None:
        assert "സെപ്റ്റംബർ" in expand("10/09/2026")

    def test_month_first_when_configured(self) -> None:
        assert "ഒക്ടോബർ" in expand("10/09/2026", ExpansionConfig(day_first=False))

    def test_clock_time(self) -> None:
        out = expand("10:30 am")
        assert "മണി" in out
        assert "മിനിറ്റ്" in out
        assert "രാവിലെ" in out

    def test_long_digit_run_is_read_digit_wise(self) -> None:
        out = expand("9847012345")
        assert out == digit_by_digit("9847012345")
        assert len(out.split()) == 10  # one word per digit, not a positional reading

    def test_short_digit_run_is_positional(self) -> None:
        assert expand("25") == "ഇരുപത്തിയഞ്ച്"

    def test_identifier_threshold_is_configurable(self) -> None:
        config = ExpansionConfig(identifier_digit_threshold=2)
        assert expand("25", config) == "രണ്ട് അഞ്ച്"

    def test_international_phone_number(self) -> None:
        assert expand("+91 9847012345").startswith("പ്ലസ്")

    def test_dates_win_over_bare_integers(self) -> None:
        """Ordering regression: a date must not decompose into three numbers."""
        assert "പത്ത് പൂജ്യം" not in expand("10/09/2026")

    def test_units_can_be_disabled(self) -> None:
        assert "km" in expand("5 km", ExpansionConfig(expand_units=False))

    def test_abbreviations_can_be_disabled(self) -> None:
        assert "ഡോ." in expand("ഡോ. രാജൻ", ExpansionConfig(expand_abbreviations=False))

    def test_celsius(self) -> None:
        assert "ഡിഗ്രി സെൽഷ്യസ്" in expand("36.6 °C")

    def test_no_digits_survive_expansion(self) -> None:
        for source in ["₹250", "50%", "2026-09-10", "10:30", "1,25,000", "5 km", "25ാം"]:
            assert not any(ch.isdigit() for ch in expand(source)), source
