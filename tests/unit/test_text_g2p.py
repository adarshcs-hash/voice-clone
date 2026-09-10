"""Grapheme-to-phoneme conversion.

The expected phoneme sequences are claims about Malayalam pronunciation. Each
test names the rule it pins.
"""

from __future__ import annotations

import pytest

from mlvoice.text.g2p import (
    PHONEME_TABLE,
    G2PConfig,
    Notation,
    phonemize,
    phonemize_word,
    to_string,
)


class TestSamvruthokaram:
    def test_final_chandrakkala_becomes_half_u(self) -> None:
        assert phonemize_word("നാട്") == ["n", "aː", "ʈ", "ɨ"]

    def test_final_u_sign_becomes_half_u(self) -> None:
        assert phonemize_word("അതു") == ["a", "t̪", "ɨ"]

    def test_non_final_u_stays_a_full_vowel(self) -> None:
        assert "u" in phonemize_word("ഉടനെ")

    def test_can_be_disabled(self) -> None:
        tokens = phonemize_word("നാട്", G2PConfig(samvruthokaram=False))
        assert tokens[-1] == "ʈ"


class TestGemination:
    def test_doubled_consonant_becomes_long(self) -> None:
        assert phonemize_word("അമ്മ") == ["a", "mː", "a"]

    def test_dental_geminate(self) -> None:
        assert phonemize_word("പത്ത്") == ["p", "a", "t̪ː", "ɨ"]

    def test_retroflex_geminate(self) -> None:
        assert "ʈː" in phonemize_word("നാട്ടിൽ")

    def test_can_be_disabled(self) -> None:
        tokens = phonemize_word("അമ്മ", G2PConfig(geminate_clusters=False))
        assert tokens.count("m") == 2


class TestSpecialClusters:
    def test_rra_is_a_geminate_alveolar_stop(self) -> None:
        """``റ്റ`` is [tː], not a doubled trill."""
        assert phonemize_word("വറ്റ്") == ["ʋ", "a", "tː", "ɨ"]

    def test_nta(self) -> None:
        """``ന്റ`` is [nd̪], not [nr]."""
        assert phonemize_word("എന്റെ") == ["e", "n", "d̪", "e"]


class TestNasalRules:
    def test_anusvara_assimilates_before_velar(self) -> None:
        assert phonemize_word("സംഗീതം")[:4] == ["s", "a", "ŋ", "ɡ"]

    def test_anusvara_stays_m_word_finally(self) -> None:
        assert phonemize_word("മലയാളം")[-1] == "m"

    @pytest.mark.parametrize(
        ("word", "nasal", "stop"),
        [("മുണ്ട്", "ɳ", "ɖ"), ("പഞ്ചസാര", "ɲ", "dʒ"), ("അമ്പലം", "m", "b")],
    )
    def test_post_nasal_voicing(self, word: str, nasal: str, stop: str) -> None:
        tokens = phonemize_word(word)
        assert nasal in tokens
        assert stop in tokens

    def test_post_nasal_voicing_can_be_disabled(self) -> None:
        tokens = phonemize_word("മുണ്ട്", G2PConfig(post_nasal_voicing=False))
        assert "ʈ" in tokens
        assert "ɖ" not in tokens

    def test_dental_nasal_cluster(self) -> None:
        """``ന്ത`` is [nd̪]; the toponym is the canonical example."""
        tokens = phonemize_word("തിരുവനന്തപുരം")
        assert "d̪" in tokens


class TestChilluAndVowels:
    @pytest.mark.parametrize(
        ("word", "final"),
        [("അവൻ", "n"), ("അവൾ", "ɭ"), ("കാർ", "r"), ("പാൽ", "l")],
    )
    def test_chillu_is_a_coda_consonant(self, word: str, final: str) -> None:
        assert phonemize_word(word)[-1] == final

    def test_inherent_a_surfaces(self) -> None:
        assert phonemize_word("മല") == ["m", "a", "l", "a"]

    def test_long_vowels_are_distinct(self) -> None:
        assert phonemize_word("കാ") == ["k", "aː"]
        assert phonemize_word("ക") == ["k", "a"]

    def test_zha_has_its_own_phoneme(self) -> None:
        assert "ɻ" in phonemize_word("കോഴി")

    def test_ra_and_rra_are_distinct(self) -> None:
        assert phonemize_word("വര") != phonemize_word("വറ")


class TestIntervocalicVoicing:
    def test_off_by_default(self) -> None:
        assert "t̪" in phonemize_word("അതു")

    def test_on_when_enabled(self) -> None:
        assert phonemize_word("അതു", G2PConfig(intervocalic_voicing=True)) == ["a", "d̪", "ɨ"]

    def test_does_not_apply_word_initially(self) -> None:
        tokens = phonemize_word("കാല്", G2PConfig(intervocalic_voicing=True))
        assert tokens[0] == "k"


class TestRendering:
    def test_ascii_symbols_exist_for_every_phoneme(self) -> None:
        """Guards against the rule tables and the symbol table drifting apart."""
        words = [
            "നാട്",
            "പത്ത്",
            "എന്റെ",
            "അമ്മ",
            "സംഗീതം",
            "മുണ്ട്",
            "വറ്റ്",
            "കോഴിക്കോട്",
            "തിരുവനന്തപുരം",
            "ജ്ഞാനം",
            "ക്ഷേത്രം",
            "ഔഷധം",
            "പഞ്ചസാര",
            "ഭഗവദ്ഗീത",
            "അവൾ",
            "കാർ",
            "പാൽ",
            "ൿ",
            "ഇരുപത്തിയഞ്ച്",
            "ഹൃദയം",
        ]
        for word in words:
            tokens = phonemize(word, G2PConfig(intervocalic_voicing=True))
            # Raises KeyError if a phoneme has no ASCII identifier.
            assert to_string(tokens, notation=Notation.ASCII)

    def test_length_mark_renders_as_colon(self) -> None:
        assert to_string([["mː"]], notation=Notation.ASCII) == "m:"

    def test_word_boundary_token(self) -> None:
        rendered = to_string(phonemize("മല കാ"))
        assert " | " in rendered

    def test_phoneme_table_has_no_duplicate_symbols(self) -> None:
        symbols = [v for k, v in PHONEME_TABLE.items() if k != "|"]
        assert len(symbols) == len(set(symbols)), "ASCII symbols must be unique"

    def test_non_malayalam_is_skipped(self) -> None:
        assert phonemize_word("work") == []
        assert phonemize("ഞാൻ work") == [["ɲ", "aː", "n"], []]

    def test_empty_input(self) -> None:
        assert phonemize_word("") == []
        assert to_string([]) == ""
