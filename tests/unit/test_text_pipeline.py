"""Chunking and the assembled text frontend."""

from __future__ import annotations

import pytest

from mlvoice.errors import TextTooLongError, ValidationError
from mlvoice.text.chunker import BreakStrength, ChunkConfig, chunk
from mlvoice.text.g2p import Notation
from mlvoice.text.pipeline import TextPipeline, TextPipelineConfig


class TestChunking:
    def test_sentences_are_separated(self) -> None:
        config = ChunkConfig(max_chars=40, min_chars=1, pack_sentences=False)
        chunks = chunk("ഒന്ന്. രണ്ട്.", config)
        assert [c.text for c in chunks] == ["ഒന്ന്.", "രണ്ട്."]

    def test_break_strengths_are_assigned(self) -> None:
        config = ChunkConfig(max_chars=40, min_chars=1, pack_sentences=False)
        chunks = chunk("ഒന്ന്. രണ്ട്.", config)
        assert chunks[0].break_after is BreakStrength.SENTENCE
        assert chunks[-1].break_after is BreakStrength.NONE

    def test_paragraph_break_is_stronger_than_sentence(self) -> None:
        chunks = chunk("ഒന്ന്.\n\nരണ്ട്.", ChunkConfig(max_chars=40, min_chars=1))
        assert chunks[0].break_after is BreakStrength.PARAGRAPH
        assert chunks[0].pause_ms > 400

    def test_a_long_word_is_never_split(self) -> None:
        word = "അന്താരാഷ്ട്രവിമാനത്താവളത്തിലേക്കുള്ളവഴിയിലൂടെപോകുന്നവർ"
        chunks = chunk(word, ChunkConfig(max_chars=20, min_chars=5))
        assert [c.text for c in chunks] == [word]

    def test_long_sentence_splits_at_clause_boundaries(self) -> None:
        text = "ഒന്നാമത്തെ ഭാഗം ഇവിടെ, രണ്ടാമത്തെ ഭാഗം ഇവിടെ, മൂന്നാമത്തെ ഭാഗം ഇവിടെ."
        chunks = chunk(text, ChunkConfig(max_chars=30, min_chars=5))
        assert len(chunks) > 1
        assert all(len(c.text) <= 40 for c in chunks)
        assert any(c.break_after is BreakStrength.CLAUSE for c in chunks)

    def test_short_fragments_are_merged(self) -> None:
        chunks = chunk("ഒന്ന്. രണ്ട്. മൂന്ന്.", ChunkConfig(max_chars=100, min_chars=50))
        assert len(chunks) == 1

    def test_indices_are_sequential(self) -> None:
        chunks = chunk("ഒന്ന്. രണ്ട്. മൂന്ന്.", ChunkConfig(max_chars=12, min_chars=1))
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_no_content_is_lost(self) -> None:
        text = "ഒന്ന്. രണ്ട് മൂന്ന് നാല്, അഞ്ച് ആറ്. ഏഴ്."
        chunks = chunk(text, ChunkConfig(max_chars=15, min_chars=1))
        rejoined = "".join(c.text for c in chunks).replace(" ", "")
        assert rejoined == text.replace(" ", "")

    def test_empty_input(self) -> None:
        assert chunk("") == []

    def test_invalid_config_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChunkConfig(max_chars=10, min_chars=10)
        with pytest.raises(ValidationError):
            ChunkConfig(max_chars=0, min_chars=0)


class TestInitials:
    """``ഒ. ജെ ജനീഷ്`` is one name, not three sentences. Initials are
    near-universal in Malayalam names in news and social copy, and splitting at
    them puts a hard stop mid-name and hands the fragment its own sentence
    prosody."""

    @pytest.mark.parametrize(
        "text",
        [
            "വളരെ പ്രശസ്തനായ കായിക വകുപ്പ് മന്ത്രി ആയിരിക്കുന്ന ഒ. ജെ ജനീഷാണ് ഉദ്ഘാടനം ചെയ്തത്.",
            "ചടങ്ങിൽ പ്രധാന അതിഥിയായി പങ്കെടുത്തത് ബഹുമാനപ്പെട്ട വി. ഡി സതീശൻ ആയിരുന്നു എന്ന് അറിഞ്ഞു.",
            "അദ്ദേഹത്തിന്റെ പേര് വളരെ വ്യക്തമായി രേഖപ്പെടുത്തിയിരിക്കുന്നത് കെ. ആർ നാരായണൻ എന്നാണ്.",
        ],
    )
    def test_an_initial_does_not_end_a_sentence(self, text: str) -> None:
        assert len(chunk(text, ChunkConfig(pack_sentences=False))) == 1

    def test_a_short_word_still_ends_a_sentence(self) -> None:
        """``ശരി`` is two letters, not one, so this is a real boundary."""
        config = ChunkConfig(max_chars=20, min_chars=1, pack_sentences=False)
        assert len(chunk("ഇത് ശരി. അതെ ശരി.", config)) == 2

    def test_latin_initials_too(self) -> None:
        config = ChunkConfig(pack_sentences=False)
        assert len(chunk("The report by A. B. Nair was published today.", config)) == 1


class TestSentencePacking:
    """A reference-prompt model re-synthesises its reference clip for every
    generation and discards it, so each extra chunk costs the reference's
    duration again. Packing is the difference between paying it five times and
    ten."""

    def test_consecutive_sentences_are_packed(self) -> None:
        text = "ഒന്നാമത്തെ വാക്യം. രണ്ടാമത്തെ വാക്യം. മൂന്നാമത്തെ വാക്യം."
        packed = chunk(text, ChunkConfig(max_chars=220, min_chars=1))
        unpacked = chunk(text, ChunkConfig(max_chars=220, min_chars=1, pack_sentences=False))
        assert len(packed) == 1
        assert len(unpacked) == 3

    def test_packing_respects_max_chars(self) -> None:
        text = " ".join(f"വാക്യം നമ്പർ {i} ഇവിടെ അവസാനിക്കുന്നു." for i in range(12))
        for c in chunk(text, ChunkConfig(max_chars=100, min_chars=1)):
            assert len(c.text) <= 100

    def test_packing_never_crosses_a_paragraph_break(self) -> None:
        chunks = chunk("ഒന്ന്.\n\nരണ്ട്.", ChunkConfig(max_chars=220, min_chars=1))
        assert len(chunks) == 2
        assert chunks[0].break_after is BreakStrength.PARAGRAPH

    def test_packing_preserves_the_final_break(self) -> None:
        chunks = chunk("ഒന്ന്. രണ്ട്.", ChunkConfig(max_chars=220, min_chars=1))
        assert chunks[-1].break_after is BreakStrength.NONE

    def test_packing_loses_no_content(self) -> None:
        text = "ഒന്ന്. രണ്ട് മൂന്ന്. നാല്, അഞ്ച്. ആറ്."
        joined = "".join(c.text for c in chunk(text, ChunkConfig(max_chars=220, min_chars=1)))
        assert joined.replace(" ", "") == text.replace(" ", "")


class TestPipeline:
    def test_stages_are_applied_in_order(self) -> None:
        result = TextPipeline().process("എൻറെ വീട്ടിൽ ₹250 ഉണ്ട്")
        assert result.normalized.startswith("എന്റെ")  # normalisation ran first
        assert "ഇരുനൂറ്റിയമ്പത് രൂപ" in result.expanded  # then expansion
        assert result.routed == result.expanded  # nothing to route here

    def test_manglish_reaches_malayalam_script(self) -> None:
        assert "ഞാൻ" in TextPipeline().process("njan veedu poyi").routed

    def test_phonemes_are_produced_per_chunk(self) -> None:
        result = TextPipeline().process("ഞാൻ നാട്ടിൽ പോയി.")
        assert len(result.phonemes) == len(result.chunks)
        assert result.phoneme_string(0)

    def test_ascii_notation(self) -> None:
        result = TextPipeline().process("നാട്")
        assert result.phoneme_string(0, notation=Notation.ASCII) == "n aa tt ax"

    def test_chunk_index_is_validated(self) -> None:
        result = TextPipeline().process("നാട്")
        with pytest.raises(ValidationError):
            result.phoneme_string(99)

    def test_summary_is_log_safe(self) -> None:
        summary = TextPipeline().process("എൻറെ വീട്").summary()
        assert summary["legacy_nta_fixed"] == 1
        assert summary["contains_malayalam"] is True
        assert set(summary) == {
            "chars_in",
            "chars_out",
            "chunks",
            "normalization_changes",
            "legacy_nta_fixed",
            "contains_malayalam",
        }

    def test_empty_text_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TextPipeline().process("   ")

    def test_over_long_text_is_rejected(self) -> None:
        pipeline = TextPipeline(TextPipelineConfig(max_chars=10))
        with pytest.raises(TextTooLongError):
            pipeline.process("ഒന്ന് രണ്ട് മൂന്ന് നാല് അഞ്ച്")

    def test_pipeline_is_reusable_and_stateless(self) -> None:
        pipeline = TextPipeline()
        first = pipeline.process("എൻറെ വീട്").routed
        pipeline.process("വേറൊരു വാക്യം")
        assert pipeline.process("എൻറെ വീട്").routed == first

    def test_plain_english_input_is_flagged_and_untouched(self) -> None:
        result = TextPipeline().process("hello world")
        assert result.routed == "hello world"
        assert result.contains_malayalam is False

    def test_text_for_synthesis_is_the_routed_form(self) -> None:
        result = TextPipeline().process("എൻറെ വീട്")
        assert result.text_for_synthesis == result.routed
