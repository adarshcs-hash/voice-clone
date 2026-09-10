"""Manglish transliteration and code-mix routing."""

from __future__ import annotations

import pytest

from mlvoice.text.codemix import (
    CodeMixConfig,
    LatinPolicy,
    ManglishDetection,
    Script,
    is_acronym,
    route,
    segment,
    spell_acronym,
)
from mlvoice.text.translit import is_probably_manglish, transliterate


class TestManglishDetection:
    @pytest.mark.parametrize(
        "word",
        ["njan", "paranju", "vannu", "kozhikode", "enikku", "cheyyunnu", "veettil"],
    )
    def test_words_carrying_a_malayalam_signal_are_detected(self, word: str) -> None:
        assert is_probably_manglish(word)

    @pytest.mark.parametrize("word", ["poyi", "veedu", "undu", "aanu", "alle"])
    def test_lexicon_membership_counts_as_evidence(self, word: str) -> None:
        assert is_probably_manglish(word)

    @pytest.mark.parametrize(
        "word",
        [
            "innovation",
            "management",
            "thinking",
            "BJP",
            "work",
            "the",
            "please",
            "world",
            "hello",
            "method",
        ],
    )
    def test_english_and_acronyms_are_rejected(self, word: str) -> None:
        assert not is_probably_manglish(word)

    def test_permissive_mode_accepts_unsignalled_words(self) -> None:
        assert not is_probably_manglish("thakarnnu"[:4])
        assert is_probably_manglish("thak", require_evidence=False)

    def test_vowelless_token_is_rejected(self) -> None:
        assert not is_probably_manglish("xyz")

    def test_non_ascii_is_rejected(self) -> None:
        assert not is_probably_manglish("ഞാൻ")


class TestTransliteration:
    def test_lexicon_pins_common_words(self) -> None:
        assert transliterate("njan") == "ഞാൻ"
        assert transliterate("veedu") == "വീട്"

    def test_lexicon_can_be_bypassed(self) -> None:
        assert transliterate("njan", use_lexicon=False) != "ഞാൻ"

    def test_sentence(self) -> None:
        assert transliterate("njan veedu poyi") == "ഞാൻ വീട് പോയി"

    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("aanu", "ആണ്"),
            ("aano", "ആണോ"),
            ("alle", "അല്ലേ"),
            ("ille", "ഇല്ലേ"),
            ("venam", "വേണം"),
            ("venda", "വേണ്ട"),
        ],
    )
    def test_copula_and_negation_family(self, word: str, expected: str) -> None:
        """The rules give a dental ``ന`` where Malayalam has retroflex ``ണ``,
        and a short ``എ`` where it has long ``ഏ``, so these are pinned."""
        assert transliterate(word) == expected

    def test_word_final_consonant_takes_chillu(self) -> None:
        assert transliterate("avan", use_lexicon=False).endswith("ൻ")

    def test_word_final_m_becomes_anusvara(self) -> None:
        assert transliterate("malayalam", use_lexicon=False).endswith("ം")

    def test_gemination_is_carried_through(self) -> None:
        assert "ക്ക" in transliterate("akkam", use_lexicon=False)

    def test_capitalised_proper_noun_is_scanned_lowercased(self) -> None:
        """A capital R is orthographic here, not the Mozhi retroflex marker."""
        out = transliterate("Rajan", use_lexicon=False, require_evidence=False)
        assert out.startswith("ര")

    def test_english_survives_untouched(self) -> None:
        assert "work" in transliterate("njan work cheythu")

    def test_plain_english_is_left_alone(self) -> None:
        assert transliterate("hello world") == "hello world"

    def test_malayalam_passes_through(self) -> None:
        assert transliterate("ഞാൻ പോയി") == "ഞാൻ പോയി"

    def test_digits_and_punctuation_pass_through(self) -> None:
        assert transliterate("veedu, 25!") == "വീട്, 25!"


class TestSegmentation:
    def test_scripts_are_separated(self) -> None:
        runs = segment("ഞാൻ work 5")
        assert [r.script for r in runs] == [
            Script.MALAYALAM,
            Script.OTHER,
            Script.LATIN,
            Script.OTHER,
            Script.DIGIT,
        ]

    def test_segmentation_is_lossless(self) -> None:
        text = "ഞാൻ work ചെയ്തു 5 MB, ok!"
        assert "".join(r.text for r in segment(text)) == text

    def test_offsets_are_correct(self) -> None:
        for run in segment("ഞാൻ work"):
            assert run.text == "ഞാൻ work"[run.start : run.end]

    def test_empty_input(self) -> None:
        assert segment("") == []


class TestAcronyms:
    @pytest.mark.parametrize("word", ["BJP", "UPSC", "PDF", "MB", "U.P.S.C"])
    def test_detected(self, word: str) -> None:
        assert is_acronym(word)

    @pytest.mark.parametrize("word", ["Kerala", "work", "Rajan", "A"])
    def test_not_detected(self, word: str) -> None:
        assert not is_acronym(word)

    def test_spelled_with_malayalam_letter_names(self) -> None:
        assert spell_acronym("BJP") == "ബി ജെ പി"

    def test_dots_are_ignored(self) -> None:
        assert spell_acronym("U.P.S.C") == spell_acronym("UPSC")


class TestRouting:
    def test_acronyms_are_spelled_out(self) -> None:
        assert "ബി ജെ പി" in route("BJP വന്നു")

    def test_manglish_is_transliterated(self) -> None:
        assert "ഞാൻ" in route("njan വന്നു")

    def test_english_is_kept_by_default(self) -> None:
        assert route("ഞാൻ work ചെയ്തു") == "ഞാൻ work ചെയ്തു"

    def test_english_can_be_dropped(self) -> None:
        out = route("ഞാൻ work ചെയ്തു", CodeMixConfig(latin_policy=LatinPolicy.DROP))
        assert out == "ഞാൻ ചെയ്തു"

    def test_english_can_be_spelled_out(self) -> None:
        out = route("ok", CodeMixConfig(latin_policy=LatinPolicy.SPELL_OUT, spell_acronyms=False))
        assert out == "ഒ കെ"

    def test_loanword_lexicon_wins(self) -> None:
        out = route("MG Road", CodeMixConfig(loanword_lexicon={"Road": "റോഡ്"}))
        assert out.endswith("റോഡ്")

    def test_lexicon_is_case_insensitive_fallback(self) -> None:
        out = route("road", CodeMixConfig(loanword_lexicon={"Road": "റോഡ്"}))
        assert out == "റോഡ്"

    def test_malayalam_and_digits_untouched(self) -> None:
        assert route("അഞ്ച് 25 ഉണ്ട്") == "അഞ്ച് 25 ഉണ്ട്"

    def test_evidence_mode_leaves_english_alone(self) -> None:
        assert route("hello world") == "hello world"

    def test_permissive_mode_transliterates_unsignalled_words(self) -> None:
        config = CodeMixConfig(manglish_detection=ManglishDetection.PERMISSIVE)
        # ``sadhanam`` carries no distinctive digraph, so evidence mode leaves
        # it alone and permissive mode transliterates it.
        assert route("sadhanam", CodeMixConfig()) == "sadhanam"
        assert route("sadhanam", config) != "sadhanam"

    def test_permissive_mode_still_respects_the_english_stoplist(self) -> None:
        config = CodeMixConfig(manglish_detection=ManglishDetection.PERMISSIVE)
        assert route("hello world", config) == "hello world"

    def test_detection_can_be_turned_off(self) -> None:
        config = CodeMixConfig(manglish_detection=ManglishDetection.OFF)
        assert route("njan paranju", config) == "njan paranju"


class TestPunctuationAndHyphens:
    """Regression: the Latin run pattern admits interior dots and hyphens for
    acronyms, so a sentence-final word arrived as ``aanu.`` and failed every
    letters-only check, silently skipping transliteration."""

    def test_sentence_final_word_is_still_routed(self) -> None:
        assert route("Kozhikode aanu.") == "കോഴിക്കോട് ആണ്."

    @pytest.mark.parametrize("punct", [".", ",", "!", "?", "...", "'"])
    def test_trailing_punctuation_is_preserved(self, punct: str) -> None:
        out = route(f"njan{punct}")
        assert out.startswith("ഞാൻ")
        assert out.endswith(punct)

    def test_acronym_with_a_trailing_stop(self) -> None:
        assert route("UPSC.") == "യു പി എസ് സി."

    def test_acronym_with_interior_dots_survives(self) -> None:
        assert route("U.P.S.C") == "യു പി എസ് സി"

    def test_hyphenated_stem_and_suffix_are_routed_separately(self) -> None:
        """Code-mixed Malayalam attaches Malayalam suffixes to English stems."""
        assert route("ഈ app-il login ചെയ്യണം") == "ഈ app-ഇൽ login ചെയ്യണം"

    def test_hyphen_is_preserved(self) -> None:
        assert route("veedu-il") == "വീട്-ഇൽ"

    def test_english_before_punctuation_is_still_left_alone(self) -> None:
        assert route("hello world.") == "hello world."

    def test_lexicon_may_pin_a_punctuated_form(self) -> None:
        config = CodeMixConfig(loanword_lexicon={"No.": "നമ്പർ"})
        assert route("No. 5", config) == "നമ്പർ 5"
