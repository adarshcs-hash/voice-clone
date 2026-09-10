"""The Malayalam hard test set.

Before training anything, measure. This set is the instrument: a fixed list of
Malayalam sentences chosen because each one probes a specific way a Malayalam
voice goes wrong, grouped so that a score can be read as a diagnosis rather than
a single number.

Run it against every system under consideration -- commercial APIs included --
score the results with native listeners, and the gaps tell you what to build.
That measurement, not a model choice, is the first task on this project.

Categories
----------
Each :class:`TestCase` names the phenomenon it probes, so a regression is
attributable. A model can pass ``NUMBERS`` and fail ``SAMVRUTHOKARAM``, and
those imply completely different work.

Size
----
The set shipped here is a **seed of ~70 cases**, hand-written to cover every
category at least twice. A production gate wants 300 or more, with at least
five cases per category and per-case native-speaker judgements collected under
:func:`scoring_instructions`. Extend it in place: the ids are stable and
referenced by report history, so add new ones rather than renumbering.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

__all__ = ["TEST_CASES", "Category", "TestCase", "by_category", "scoring_instructions"]


class Category(StrEnum):
    """The phenomenon a case probes."""

    SAMVRUTHOKARAM = "samvruthokaram"
    """Word-final half-u. The most audible Malayalam-specific error."""
    GEMINATION = "gemination"
    """Doubled consonants, which are phonemically long."""
    NASAL_CLUSTER = "nasal_cluster"
    """Homorganic nasal assimilation and post-nasal voicing."""
    CHILLU = "chillu"
    """Pure-consonant letters in coda position."""
    LEGACY_ENCODING = "legacy_encoding"
    """Text carrying pre-Unicode-5.1 spellings that must normalise first."""
    PLACE_NAME = "place_name"
    """Kerala toponyms, which multilingual models routinely mangle."""
    PERSON_NAME = "person_name"
    NUMBERS = "numbers"
    DATE_TIME = "date_time"
    CURRENCY = "currency"
    CODE_MIX = "code_mix"
    """English words inside a Malayalam sentence."""
    MANGLISH = "manglish"
    """Malayalam written in Latin script."""
    ACRONYM = "acronym"
    AGGLUTINATION = "agglutination"
    """Very long single words, which break naive chunking."""
    SANSKRIT_LOAN = "sanskrit_loan"
    """Aspirates and clusters that only occur in Sanskrit vocabulary."""
    MINIMAL_PAIR = "minimal_pair"
    """Contrasts a listener will notice instantly if collapsed."""
    PROSODY = "prosody"
    """Questions, exclamations and lists, where intonation carries meaning."""
    LONG_FORM = "long_form"
    """Multi-sentence input, which exposes prosody drift and chunk seams."""


@dataclass(frozen=True, slots=True)
class TestCase:
    """One evaluation item.

    Attributes:
        id: Stable identifier. Never renumber; reports are keyed on it.
        text: The input, exactly as a caller would send it.
        category: What this case probes.
        probes: What a listener or scorer should be checking.
        expected_text: For cases where the text frontend has one correct
            output, the expected normalised form. Lets the frontend be gated
            automatically, with no audio and no listener.
    """

    id: str
    text: str
    category: Category
    probes: str
    expected_text: str | None = None


C = Category

TEST_CASES: Final[tuple[TestCase, ...]] = (
    # -- samvruthokaram ----------------------------------------------------
    TestCase("sam-001", "അവൻ നാട്ടിൽ എത്തി.", C.SAMVRUTHOKARAM, "final ്‍ in നാട്ടിൽ; no full [u]"),
    TestCase(
        "sam-002", "ഇത് എന്റെ വീട് ആണ്.", C.SAMVRUTHOKARAM, "three word-final chandrakkala in a row"
    ),
    TestCase("sam-003", "അതു ശരിയാണു.", C.SAMVRUTHOKARAM, "final ു must be [ɨ], not [u]"),
    TestCase("sam-004", "പാലക്കാട് വഴി പോയി.", C.SAMVRUTHOKARAM, "place name ending in ്‍"),
    TestCase("sam-005", "കുട്ടിക്ക് പുസ്തകം കൊടുത്തു.", C.SAMVRUTHOKARAM, "dative ക്ക് plus final ു"),
    # -- gemination --------------------------------------------------------
    TestCase("gem-001", "അമ്മ വന്നു.", C.GEMINATION, "മ്മ length"),
    TestCase("gem-002", "പത്ത് രൂപ തന്നു.", C.GEMINATION, "ത്ത vs single ത"),
    TestCase("gem-003", "കൊച്ചിയിൽ മഴ പെയ്തു.", C.GEMINATION, "ച്ചി palatal geminate"),
    TestCase(
        "gem-004", "വറ്റൽ മുളക് വേണം.", C.GEMINATION, "റ്റ is a geminate stop, not a doubled trill"
    ),
    TestCase("gem-005", "ഇപ്പോൾ ഒന്നും വേണ്ട.", C.GEMINATION, "പ്പ plus ൾ coda"),
    # -- nasal clusters ----------------------------------------------------
    TestCase("nas-001", "സംഗീതം ഇഷ്ടമാണ്.", C.NASAL_CLUSTER, "ം before ഗ assimilates to [ŋ]"),
    TestCase("nas-002", "മുണ്ട് ഉടുത്തു.", C.NASAL_CLUSTER, "ണ്ട is [ɳɖ], voiced"),
    TestCase("nas-003", "പഞ്ചസാര ചേർത്തു.", C.NASAL_CLUSTER, "ഞ്ച is [ɲdʒ]"),
    TestCase("nas-004", "അമ്പലത്തിൽ പോയി.", C.NASAL_CLUSTER, "മ്പ is [mb]"),
    TestCase("nas-005", "തിരുവനന്തപുരം ജില്ല.", C.NASAL_CLUSTER, "ന്ത is [nd̪]; also a place name"),
    # -- chillu ------------------------------------------------------------
    TestCase("chi-001", "അവൾ പറഞ്ഞു.", C.CHILLU, "ൾ coda"),
    TestCase("chi-002", "കാർ ഓടിച്ചു.", C.CHILLU, "ർ coda, not a vowel"),
    TestCase("chi-003", "പാൽ കുടിച്ചു.", C.CHILLU, "ൽ coda"),
    TestCase("chi-004", "മരണം ഒരു സത്യമാണെൻ.", C.CHILLU, "ൻ coda word-final"),
    TestCase("chi-005", "ലക്ഷ്യം നേടി.", C.CHILLU, "ക്ഷ cluster with ്യ"),
    # -- legacy encodings --------------------------------------------------
    TestCase(
        "leg-001",
        "എൻറെ വീട്",
        C.LEGACY_ENCODING,
        "chillu-n + ṟa must normalise to ന്റ",
        expected_text="എന്റെ വീട്",
    ),
    TestCase(
        "leg-002",
        "അവന്‍ പോയി",
        C.LEGACY_ENCODING,
        "ZWJ chillu must compose to ൻ",
        expected_text="അവൻ പോയി",
    ),
    TestCase(
        "leg-003",
        "മൌനം നല്ലതാണ്",
        C.LEGACY_ENCODING,
        "legacy ൌ must become ൗ",
        expected_text="മൗനം നല്ലതാണ്",
    ),
    TestCase(
        "leg-004",
        "വൎഷം കഴിഞ്ഞു",
        C.LEGACY_ENCODING,
        "dot reph must become ർ",
        expected_text="വർഷം കഴിഞ്ഞു",
    ),
    TestCase(
        "leg-005",
        "൧൨൩ രൂപ",
        C.LEGACY_ENCODING,
        "Malayalam digits must convert then expand",
        expected_text="നൂറ്റിയിരുപത്തിമൂന്ന് രൂപ",
    ),
    # -- place names -------------------------------------------------------
    TestCase("plc-001", "തിരുവനന്തപുരത്ത് നിന്ന് കോഴിക്കോട്ടേക്ക്.", C.PLACE_NAME, "two hard toponyms"),
    TestCase("plc-002", "ആലപ്പുഴ, കാസർകോട്, പത്തനംതിട്ട.", C.PLACE_NAME, "list intonation plus toponyms"),
    TestCase("plc-003", "തൃശ്ശൂർ പൂരം കണ്ടു.", C.PLACE_NAME, "ൃ plus ശ്ശ geminate"),
    TestCase("plc-004", "എറണാകുളം ജംഗ്ഷനിൽ ഇറങ്ങി.", C.PLACE_NAME, "ംഗ്ഷ cluster in a loanword"),
    TestCase("plc-005", "വയനാട്ടിലെ മലനിരകൾ.", C.PLACE_NAME, "ട്ടി geminate in a toponym"),
    # -- person names ------------------------------------------------------
    TestCase("per-001", "ശ്രീ രാജൻ നായർ എത്തി.", C.PERSON_NAME, "honorific plus name"),
    TestCase("per-002", "ഡോ. ഫാത്തിമ ബീവി സംസാരിച്ചു.", C.PERSON_NAME, "abbreviation plus loan name"),
    TestCase("per-003", "കുഞ്ഞുണ്ണി മാഷ് കവിത എഴുതി.", C.PERSON_NAME, "ഞ്ഞ and ണ്ണ geminates"),
    # -- numbers -----------------------------------------------------------
    TestCase(
        "num-001", "25 പേർ വന്നു.", C.NUMBERS, "tens+units sandhi", expected_text="ഇരുപത്തിയഞ്ച് പേർ വന്നു."
    ),
    TestCase(
        "num-002",
        "1,25,000 രൂപ",
        C.NUMBERS,
        "lakh grouping",
        expected_text="ഒരു ലക്ഷത്തി ഇരുപത്തിയയ്യായിരം രൂപ",
    ),
    TestCase(
        "num-003", "3 കോടി ജനങ്ങൾ", C.NUMBERS, "crore multiplier", expected_text="മൂന്ന് കോടി ജനങ്ങൾ"
    ),
    TestCase("num-004", "3.75 ശതമാനം", C.NUMBERS, "decimal read digit-wise after the point"),
    TestCase("num-005", "25ാം തീയതി", C.NUMBERS, "ordinal", expected_text="ഇരുപത്തിയഞ്ചാം തീയതി"),
    TestCase(
        "num-006", "എന്റെ നമ്പർ 9847012345", C.NUMBERS, "phone number must be read digit by digit"
    ),
    TestCase("num-007", "8000 പേർ പങ്കെടുത്തു.", C.NUMBERS, "irregular thousand form എണ്ണായിരം"),
    TestCase("num-008", "10000 രൂപ വേണം.", C.NUMBERS, "irregular പതിനായിരം"),
    # -- dates and times ---------------------------------------------------
    TestCase("dat-001", "2026-09-10 ന് യോഗം.", C.DATE_TIME, "ISO date"),
    TestCase("dat-002", "10/09/2026 ന് വരാം.", C.DATE_TIME, "day-first date"),
    TestCase("dat-003", "രാവിലെ 10:30 ന് തുടങ്ങും.", C.DATE_TIME, "clock time"),
    TestCase("dat-004", "വൈകുന്നേരം 6 മണിക്ക്.", C.DATE_TIME, "bare hour"),
    # -- currency ----------------------------------------------------------
    TestCase(
        "cur-001",
        "₹250 തന്നു.",
        C.CURRENCY,
        "symbol becomes a following word",
        expected_text="ഇരുനൂറ്റിയമ്പത് രൂപ തന്നു.",
    ),
    TestCase("cur-002", "Rs. 5000 ആണ് വില.", C.CURRENCY, "Latin abbreviation"),
    TestCase("cur-003", "$100 ഡോളർ അല്ല.", C.CURRENCY, "foreign currency symbol"),
    # -- code mix ----------------------------------------------------------
    TestCase("cod-001", "എനിക്ക് ഒരു meeting ഉണ്ട്.", C.CODE_MIX, "single English noun mid-sentence"),
    TestCase("cod-002", "PDF file 5 MB ആണ്.", C.CODE_MIX, "acronym plus unit plus English noun"),
    TestCase(
        "cod-003", "ഞാൻ work from home ചെയ്യുന്നു.", C.CODE_MIX, "English phrase inside Malayalam syntax"
    ),
    TestCase(
        "cod-004",
        "ഈ app-il login ചെയ്യണം.",
        C.CODE_MIX,
        "hyphenated Malayalam suffix on an English stem",
    ),
    # -- manglish ----------------------------------------------------------
    TestCase("man-001", "njan veedu poyi", C.MANGLISH, "core Manglish must reach Malayalam script"),
    TestCase("man-002", "enikku vishakkunnu", C.MANGLISH, "geminate written double in Latin"),
    TestCase("man-003", "Kozhikode ninnu vannu", C.MANGLISH, "capitalised toponym in Latin"),
    # -- acronyms ----------------------------------------------------------
    TestCase("acr-001", "UPSC പരീക്ഷ എഴുതി.", C.ACRONYM, "spell out with Malayalam letter names"),
    TestCase("acr-002", "KSRTC ബസ് വന്നു.", C.ACRONYM, "five-letter acronym"),
    TestCase(
        "acr-003", "ISRO വിക്ഷേപണം നടത്തി.", C.ACRONYM, "acronym that reads as a word in English"
    ),
    # -- agglutination -----------------------------------------------------
    TestCase("agg-001", "വീട്ടിലേക്കായിരുന്നു അവൻ പോയത്.", C.AGGLUTINATION, "long inflected verb form"),
    TestCase(
        "agg-002", "അന്താരാഷ്ട്രവിമാനത്താവളത്തിലേക്ക് പോകുന്നു.", C.AGGLUTINATION, "38-character single word"
    ),
    TestCase(
        "agg-003", "പറഞ്ഞുകൊണ്ടിരിക്കുന്നവരോട് സംസാരിച്ചു.", C.AGGLUTINATION, "stacked verbal morphology"
    ),
    # -- Sanskrit loans ----------------------------------------------------
    TestCase("san-001", "ജ്ഞാനം ശക്തിയാണ്.", C.SANSKRIT_LOAN, "ജ്ഞ cluster"),
    TestCase("san-002", "ഭഗവദ്ഗീത വായിച്ചു.", C.SANSKRIT_LOAN, "voiced aspirates ഭ and ദ്ഗ"),
    TestCase("san-003", "ക്ഷേത്രത്തിൽ ദർശനം നടത്തി.", C.SANSKRIT_LOAN, "ക്ഷ plus ർശ"),
    TestCase("san-004", "ഔഷധം കഴിച്ചു.", C.SANSKRIT_LOAN, "initial ഔ diphthong"),
    # -- minimal pairs -----------------------------------------------------
    TestCase("min-001", "കളം കഴം കലം", C.MINIMAL_PAIR, "ള vs ഴ vs ല must stay distinct"),
    TestCase("min-002", "പട്ടി പതി പഠി", C.MINIMAL_PAIR, "ട്ട vs ത vs ഠ"),
    TestCase("min-003", "വര വറ", C.MINIMAL_PAIR, "ര [ɾ] vs റ [r]"),
    TestCase(
        "min-004", "കാൽ കാല്", C.MINIMAL_PAIR, "chillu vs chandrakkala spelling of the same word"
    ),
    # -- prosody -----------------------------------------------------------
    TestCase("pro-001", "നിങ്ങൾ എവിടെ പോകുന്നു?", C.PROSODY, "wh-question contour"),
    TestCase("pro-002", "അയ്യോ! എന്തൊരു കഷ്ടം!", C.PROSODY, "exclamation"),
    TestCase("pro-003", "ചായ, കാപ്പി, പാൽ, വെള്ളം — എന്ത് വേണം?", C.PROSODY, "list then question"),
    TestCase("pro-004", "അവൻ വന്നു, പക്ഷേ ഞാൻ പോയി.", C.PROSODY, "contrastive clause break"),
    # -- long form ---------------------------------------------------------
    TestCase(
        "lng-001",
        "കേരളം ഇന്ത്യയുടെ തെക്കുപടിഞ്ഞാറൻ തീരത്തുള്ള ഒരു സംസ്ഥാനമാണ്. "
        "1956 നവംബർ ഒന്നിനാണ് ഇത് രൂപീകരിച്ചത്. "
        "മലയാളമാണ് ഇവിടത്തെ ഭരണഭാഷ. "
        "സാക്ഷരതയിൽ ഇന്ത്യയിൽ ഒന്നാം സ്ഥാനത്താണ് കേരളം.",
        C.LONG_FORM,
        "four sentences: chunk seams, prosody drift, a year, an ordinal",
    ),
    TestCase(
        "lng-002",
        "ഇന്നത്തെ കാലാവസ്ഥാ പ്രവചനം: കൊച്ചിയിൽ 32 ഡിഗ്രി വരെ ചൂട് ഉയരും. "
        "വൈകുന്നേരം 4 മണിക്ക് ശേഷം മഴയ്ക്ക് 60 ശതമാനം സാധ്യതയുണ്ട്. "
        "മത്സ്യബന്ധനത്തിന് പോകുന്നവർ ജാഗ്രത പാലിക്കണം.",
        C.LONG_FORM,
        "bulletin register with numbers, a temperature and a percentage",
    ),
)


def by_category(category: Category) -> tuple[TestCase, ...]:
    """Return every case in ``category``."""
    return tuple(case for case in TEST_CASES if case.category is category)


def iter_cases(categories: Sequence[Category] | None = None) -> Iterator[TestCase]:
    """Iterate cases, optionally restricted to ``categories``."""
    if categories is None:
        yield from TEST_CASES
        return
    wanted = set(categories)
    yield from (case for case in TEST_CASES if case.category in wanted)


def category_counts() -> dict[str, int]:
    """Cases per category, to check coverage before trusting a report."""
    counts: dict[str, int] = {}
    for case in TEST_CASES:
        counts[case.category.value] = counts.get(case.category.value, 0) + 1
    return dict(sorted(counts.items()))


def scoring_instructions() -> str:
    """Instructions to hand to native-speaker raters.

    Objective metrics cannot see the errors that matter most here -- an ASR
    system that shares the TTS system's mistakes will happily transcribe a
    mispronounced word back to the right characters. Human scoring on these
    four axes is the ground truth this project gates on; CER is the cheap proxy
    used between rating rounds.
    """
    return (
        "Rate each clip 1-5 on four independent axes. Do not average them "
        "yourself; the report does.\n\n"
        "1. NATURALNESS - does it sound like a person speaking Malayalam, or "
        "like a machine reading letters?\n"
        "2. PRONUNCIATION - is every word pronounced the way you would say it? "
        "Mark specifically: samvruthokaram (final half-u), consonant length "
        "(gemination), and nasal+stop clusters.\n"
        "3. INTELLIGIBILITY - could you understand it first time, without "
        "reading the text?\n"
        "4. SPEAKER SIMILARITY (cloned voices only) - does it sound like the "
        "same person as the reference clip?\n\n"
        "Also flag, per clip: any word you would have said differently, and "
        "whether the accent or dialect sounds wrong for the content.\n\n"
        "Use at least five raters per clip and report the median with the "
        "inter-rater range. A single rater's score is not a measurement."
    )
