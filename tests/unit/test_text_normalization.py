"""Malayalam Unicode normalisation and the character inventory."""

from __future__ import annotations

import pytest

from mlvoice.text import chars
from mlvoice.text.unicode_norm import decompose_chillu, normalize, normalize_with_report

ZWJ = "‍"
ZWNJ = "‌"


class TestCharacterInventory:
    def test_chillu_maps_are_inverses(self) -> None:
        for chillu, base in chars.CHILLU_TO_BASE.items():
            assert chars.BASE_TO_CHILLU[base] == chillu

    def test_classification_is_exclusive(self) -> None:
        for ch in chars.CONSONANTS:
            assert chars.is_consonant(ch)
            assert not chars.is_chillu(ch)
            assert not chars.is_vowel_sign(ch)

    def test_every_chillu_is_classified(self) -> None:
        for ch in chars.CHILLUS:
            assert chars.is_chillu(ch)
            assert chars.is_malayalam(ch)

    def test_has_malayalam(self) -> None:
        assert chars.has_malayalam("ഞാൻ work")
        assert not chars.has_malayalam("plain ascii 123")

    def test_malayalam_digits_cover_zero_to_nine(self) -> None:
        assert sorted(chars.MALAYALAM_DIGITS.values()) == [str(d) for d in range(10)]


class TestChilluComposition:
    def test_zwj_sequence_becomes_atomic_chillu(self) -> None:
        assert normalize(f"അവന{chars.VIRAMA}{ZWJ} പോയി") == "അവൻ പോയി"

    @pytest.mark.parametrize(
        ("base", "chillu"),
        [("ണ", "ൺ"), ("ന", "ൻ"), ("ര", "ർ"), ("ല", "ൽ"), ("ള", "ൾ"), ("ക", "ൿ")],
    )
    def test_every_chillu_composes(self, base: str, chillu: str) -> None:
        assert normalize(f"x{base}{chars.VIRAMA}{ZWJ}") == f"x{chillu}"

    def test_decompose_round_trips(self) -> None:
        original = normalize("അവൻ കാർ ഓടിച്ചു")
        assert normalize(decompose_chillu(original)) == original


class TestLegacyRepairs:
    def test_legacy_nta_is_repaired(self) -> None:
        assert normalize("എൻറെ") == "എന്റെ"

    def test_legacy_nta_is_counted(self) -> None:
        _, report = normalize_with_report("എൻറെ വീട്, അവൻറെ വീട്")
        assert report.legacy_nta_fixed == 2
        assert report.notes

    def test_aggressive_mode_is_opt_in(self) -> None:
        # ``മുൻനിര`` is a legitimate compound; the default must not touch it.
        assert normalize("മുൻനിര") == "മുൻനിര"
        assert normalize("മുൻനിര", aggressive_legacy=True) == "മുന്നിര"

    @pytest.mark.parametrize(
        ("source", "expected"),
        [("മൌനം", "മൗനം"), ("വൎഷം", "വർഷം"), ("ൟ", "ഈ")],
    )
    def test_archaic_letters_are_mapped(self, source: str, expected: str) -> None:
        assert normalize(source) == expected


class TestZeroWidthAndDigits:
    def test_zwnj_is_stripped(self) -> None:
        assert normalize(f"ക{chars.VIRAMA}{ZWNJ}ക") == f"ക{chars.VIRAMA}ക"

    def test_stray_zwj_is_stripped(self) -> None:
        assert ZWJ not in normalize(f"മല{ZWJ}യാളം")

    def test_malayalam_digits_convert(self) -> None:
        assert normalize("൧൨൩") == "123"

    def test_digit_conversion_can_be_disabled(self) -> None:
        assert normalize("൧൨൩", convert_digits=False) == "൧൨൩"

    def test_numeral_signs_convert(self) -> None:
        assert normalize("൱") == "100"


class TestWhitespaceAndPunctuation:
    def test_whitespace_is_collapsed_and_trimmed(self) -> None:
        assert normalize("  ഒന്ന്    രണ്ട്  ") == "ഒന്ന് രണ്ട്"

    def test_paragraphs_are_preserved_as_one_blank_line(self) -> None:
        assert normalize("ഒന്ന്.\n\n\n\nരണ്ട്.") == "ഒന്ന്.\n\nരണ്ട്."

    def test_danda_becomes_a_full_stop(self) -> None:
        assert normalize("ഒന്ന്।") == "ഒന്ന്."

    def test_smart_quotes_are_folded(self) -> None:
        assert normalize("“ഒന്ന്”") == '"ഒന്ന്"'

    def test_control_characters_are_removed(self) -> None:
        text, report = normalize_with_report("ഒന്ന്\x00\x07")
        assert text == "ഒന്ന്"
        assert report.control_stripped == 2

    def test_doubled_virama_collapses(self) -> None:
        assert normalize(f"നാട{chars.VIRAMA}{chars.VIRAMA}") == f"നാട{chars.VIRAMA}"


class TestInvariants:
    @pytest.mark.parametrize(
        "text",
        [
            "എൻറെ വീട്",
            "അവന്‍ പോയി",
            "മൌനം",
            "൧൨൩ രൂപ",
            "plain ascii",
            "ഞാൻ work ചെയ്തു",
            "",
        ],
    )
    def test_normalization_is_idempotent(self, text: str) -> None:
        once = normalize(text)
        assert normalize(once) == once

    def test_report_flags_unchanged_text(self) -> None:
        _, report = normalize_with_report("ഇത് ശരി")
        assert report.changed is False
        assert report.total_changes == 0

    def test_report_total_is_the_sum_of_parts(self) -> None:
        _, report = normalize_with_report("എൻറെ ൧൨ മൌനം അവന്‍")
        assert report.total_changes == (
            report.chillu_composed
            + report.legacy_nta_fixed
            + report.aggressive_legacy_fixed
            + report.archaic_mapped
            + report.zero_width_stripped
            + report.digits_converted
            + report.control_stripped
        )
