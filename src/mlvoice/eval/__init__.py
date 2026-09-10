"""Objective metrics, the Malayalam hard test set, and the evaluation harness."""

from mlvoice.eval.metrics import (
    ErrorRate,
    character_error_rate,
    edit_distance,
    normalize_for_scoring,
    word_error_rate,
)
from mlvoice.eval.run import (
    EvaluationHarness,
    FrontendCaseResult,
    FrontendReport,
    SynthesisCaseResult,
    SynthesisReport,
)
from mlvoice.eval.speaker import EcapaSpeakerVerifier
from mlvoice.eval.testset import (
    TEST_CASES,
    Category,
    TestCase,
    by_category,
    category_counts,
    iter_cases,
    scoring_instructions,
)

__all__ = [
    "TEST_CASES",
    "Category",
    "EcapaSpeakerVerifier",
    "ErrorRate",
    "EvaluationHarness",
    "FrontendCaseResult",
    "FrontendReport",
    "SynthesisCaseResult",
    "SynthesisReport",
    "TestCase",
    "by_category",
    "category_counts",
    "character_error_rate",
    "edit_distance",
    "iter_cases",
    "normalize_for_scoring",
    "scoring_instructions",
    "word_error_rate",
]
