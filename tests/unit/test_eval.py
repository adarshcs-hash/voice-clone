"""Metrics, the hard test set, and the evaluation harness."""

from __future__ import annotations

import json

import pytest

from mlvoice.audio.io import Audio
from mlvoice.errors import ValidationError
from mlvoice.eval.metrics import (
    character_error_rate,
    edit_distance,
    normalize_for_scoring,
    word_error_rate,
)
from mlvoice.eval.run import EvaluationHarness
from mlvoice.eval.testset import (
    TEST_CASES,
    Category,
    by_category,
    category_counts,
    iter_cases,
    scoring_instructions,
)
from mlvoice.text.pipeline import TextPipeline
from mlvoice.tts.dummy import DummySynthesizer
from tests.conftest import FakeTranscriber


class TestEditDistance:
    def test_identical_sequences(self) -> None:
        assert edit_distance(["a", "b"], ["a", "b"]) == (0, 0, 0)

    def test_substitution(self) -> None:
        assert edit_distance(["a", "b"], ["a", "c"]) == (1, 0, 0)

    def test_deletion(self) -> None:
        assert edit_distance(["a", "b"], ["a"]) == (0, 1, 0)

    def test_insertion(self) -> None:
        assert edit_distance(["a"], ["a", "b"]) == (0, 0, 1)

    def test_empty_reference(self) -> None:
        assert edit_distance([], ["a", "b"]) == (0, 0, 2)

    def test_empty_hypothesis(self) -> None:
        assert edit_distance(["a", "b"], []) == (0, 2, 0)


class TestErrorRates:
    def test_perfect_match_scores_zero(self) -> None:
        assert character_error_rate("നാട്", "നാട്").rate == 0.0

    def test_normalisation_removes_spurious_errors(self) -> None:
        """A legacy spelling difference is not a pronunciation error."""
        assert character_error_rate("എന്റെ വീട്", "എൻറെ വീട്").rate == 0.0

    def test_punctuation_is_ignored_by_default(self) -> None:
        assert character_error_rate("നാട്.", "നാട്").rate == 0.0

    def test_punctuation_can_be_scored(self) -> None:
        rate = character_error_rate("നാട്.", "നാട്", strip_punctuation=False).rate
        assert rate > 0.0

    def test_word_error_rate(self) -> None:
        result = word_error_rate("ഞാൻ വീട്ടിൽ പോയി", "ഞാൻ വീട്ടിൽ വന്നു")
        assert result.rate == pytest.approx(1 / 3)
        assert result.substitutions == 1

    def test_counts_add_up(self) -> None:
        result = word_error_rate("a b c", "a x")
        assert result.errors == result.substitutions + result.deletions + result.insertions

    def test_empty_reference_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            character_error_rate("", "എന്തോ")

    def test_result_is_serialisable(self) -> None:
        payload = character_error_rate("നാട്", "നാ").as_dict()
        assert set(payload) == {
            "rate",
            "substitutions",
            "deletions",
            "insertions",
            "reference_length",
        }

    def test_scoring_normalisation_collapses_whitespace(self) -> None:
        assert normalize_for_scoring("  ഒന്ന്   രണ്ട്  ") == "ഒന്ന് രണ്ട്"

    @pytest.mark.parametrize(
        ("reference", "hypothesis"),
        [("abc def", "abc dfe"), ("hello world foo", "helo wrld"), ("a", "b")],
    )
    def test_matches_jiwer_when_available(self, reference: str, hypothesis: str) -> None:
        jiwer = pytest.importorskip("jiwer")
        assert word_error_rate(reference, hypothesis).rate == pytest.approx(
            jiwer.wer(reference, hypothesis)
        )
        assert character_error_rate(reference, hypothesis).rate == pytest.approx(
            jiwer.cer(reference.replace(" ", ""), hypothesis.replace(" ", ""))
        )


class TestTestSet:
    def test_ids_are_unique(self) -> None:
        ids = [case.id for case in TEST_CASES]
        assert len(ids) == len(set(ids))

    def test_every_category_is_covered(self) -> None:
        counts = category_counts()
        for category in Category:
            assert counts.get(category.value, 0) >= 2, f"{category.value} needs more cases"

    def test_every_case_has_text_and_a_probe(self) -> None:
        for case in TEST_CASES:
            assert case.text.strip()
            assert case.probes.strip()

    def test_by_category(self) -> None:
        cases = by_category(Category.SAMVRUTHOKARAM)
        assert cases
        assert all(case.category is Category.SAMVRUTHOKARAM for case in cases)

    def test_iter_cases_filters(self) -> None:
        selected = list(iter_cases([Category.NUMBERS, Category.CURRENCY]))
        assert {case.category for case in selected} == {Category.NUMBERS, Category.CURRENCY}

    def test_iter_cases_unfiltered_returns_everything(self) -> None:
        assert len(list(iter_cases())) == len(TEST_CASES)

    def test_scoring_instructions_name_the_axes(self) -> None:
        text = scoring_instructions()
        for axis in ("NATURALNESS", "PRONUNCIATION", "INTELLIGIBILITY", "SPEAKER SIMILARITY"):
            assert axis in text
        assert "samvruthokaram" in text

    def test_manglish_cases_reach_malayalam_script(self) -> None:
        """Regression: the Manglish cases must actually be transliterated."""
        pipeline = TextPipeline()
        for case in by_category(Category.MANGLISH):
            result = pipeline.process(case.text)
            assert result.contains_malayalam, case.id


class TestFrontendEvaluation:
    def test_all_assertions_pass(self) -> None:
        report = EvaluationHarness().evaluate_frontend()
        assert report.failures == ()
        assert report.pass_rate == 1.0

    def test_every_case_is_reported(self) -> None:
        report = EvaluationHarness().evaluate_frontend()
        assert len(report.results) == len(TEST_CASES)

    def test_asserted_cases_are_identified(self) -> None:
        report = EvaluationHarness().evaluate_frontend()
        assert len(report.asserted) == sum(1 for c in TEST_CASES if c.expected_text)

    def test_failures_are_surfaced(self) -> None:
        from mlvoice.eval.testset import TestCase

        broken = TestCase(
            "x-001", "നാട്", Category.SAMVRUTHOKARAM, "probe", expected_text="something else"
        )
        report = EvaluationHarness().evaluate_frontend([broken])
        assert report.pass_rate == 0.0
        assert report.failures[0].id == "x-001"

    def test_report_is_valid_json(self) -> None:
        payload = json.loads(EvaluationHarness().evaluate_frontend().to_json())
        assert payload["kind"] == "frontend"
        assert payload["by_category"]

    def test_per_category_counts(self) -> None:
        summary = EvaluationHarness().evaluate_frontend().by_category()
        assert summary["numbers"]["cases"] >= 5

    def test_phonemes_are_recorded(self) -> None:
        report = EvaluationHarness().evaluate_frontend(by_category(Category.SAMVRUTHOKARAM))
        assert all(result.phonemes for result in report.results)


class TestSynthesisEvaluation:
    @pytest.fixture
    def backend(self) -> DummySynthesizer:
        synthesizer = DummySynthesizer()
        synthesizer.load()
        return synthesizer

    def test_every_case_is_measured(self, backend: DummySynthesizer) -> None:
        cases = by_category(Category.SAMVRUTHOKARAM)
        report = EvaluationHarness().evaluate_synthesis(backend, cases)
        assert len(report.results) == len(cases)
        assert report.failures == ()
        assert all(result.audio_seconds > 0 for result in report.results)

    def test_cer_is_reported_with_a_transcriber(
        self, backend: DummySynthesizer, transcriber: FakeTranscriber
    ) -> None:
        transcriber.text = "അവൻ നാട്ടിൽ എത്തി"
        harness = EvaluationHarness(transcriber=transcriber)
        report = harness.evaluate_synthesis(backend, by_category(Category.SAMVRUTHOKARAM))
        assert all(result.cer is not None for result in report.results)
        assert report.to_dict()["mean_cer"] is not None

    def test_cer_is_absent_without_a_transcriber(self, backend: DummySynthesizer) -> None:
        report = EvaluationHarness().evaluate_synthesis(backend, TEST_CASES[:3])
        assert report.to_dict()["mean_cer"] is None

    def test_mos_estimator_is_used(self, backend: DummySynthesizer) -> None:
        class FixedMos:
            def score(self, audio: Audio) -> float:
                return 3.5

        report = EvaluationHarness(mos_estimator=FixedMos()).evaluate_synthesis(
            backend, TEST_CASES[:2]
        )
        assert report.to_dict()["mean_mos"] == pytest.approx(3.5)

    def test_a_failing_case_does_not_abort_the_run(self, backend: DummySynthesizer) -> None:
        class Exploding(DummySynthesizer):
            def _synthesize_chunk(self, chunk_text, request):  # type: ignore[no-untyped-def]
                raise RuntimeError("boom")

        broken = Exploding()
        broken.load()
        report = EvaluationHarness().evaluate_synthesis(broken, TEST_CASES[:3])
        assert len(report.failures) == 3
        assert report.to_dict()["errors"] == 3

    def test_category_filter(self, backend: DummySynthesizer) -> None:
        report = EvaluationHarness().evaluate_synthesis(
            backend, TEST_CASES, categories=[Category.CURRENCY]
        )
        assert {result.category for result in report.results} == {"currency"}

    def test_report_is_valid_json(self, backend: DummySynthesizer) -> None:
        payload = json.loads(
            EvaluationHarness().evaluate_synthesis(backend, TEST_CASES[:3]).to_json()
        )
        assert payload["kind"] == "synthesis"
        assert payload["backend"] == "dummy"
