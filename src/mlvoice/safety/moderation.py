"""Request and enrolment moderation.

Voice cloning has a small number of high-volume abuse patterns, and Malayalam
has its own version of each: political deepfakes around Kerala elections,
cloned voices of film actors, and above all the "relative in trouble, send money
now" phone scam, which is devastatingly effective in a language where the
listener has never heard synthetic Malayalam before.

This module is the cheap deterministic first layer:

*   :class:`PatternModerator` flags synthesis text matching known fraud
    templates in Malayalam and English.
*   :class:`NameBlocklist` refuses enrolment of voices labelled with the name of
    a public figure.

Neither is sufficient alone. A pattern list catches the lazy attempt and raises
the cost of the rest; the layers that actually stop a determined actor are
identity-verified enrolment (:mod:`mlvoice.voices.consent`), watermarking
(:mod:`mlvoice.safety.watermark`) and rate limiting. This module is written so a
trained classifier can replace :class:`PatternModerator` behind the same
protocol without touching the call sites.

False positives are a real cost: refusing a legitimate audiobook because it
contains the word "OTP" is its own failure. Decisions therefore carry a
severity, and only ``BLOCK`` refuses the request; ``FLAG`` allows it and marks
it for review.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

from mlvoice.logging import get_logger

__all__ = [
    "FRAUD_PATTERNS",
    "ModerationDecision",
    "Moderator",
    "NameBlocklist",
    "PatternModerator",
    "Severity",
]

log = get_logger(__name__)


class Severity(StrEnum):
    """What to do with a moderated request."""

    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class ModerationDecision:
    """Moderation outcome and its justification."""

    severity: Severity
    rules: tuple[str, ...] = field(default_factory=tuple)

    @property
    def blocked(self) -> bool:
        """True when the request must be refused."""
        return self.severity is Severity.BLOCK

    def as_dict(self) -> dict[str, object]:
        """Flat mapping for logs and API responses."""
        return {"severity": self.severity.value, "rules": list(self.rules)}


@runtime_checkable
class Moderator(Protocol):
    """Anything that can classify synthesis text."""

    def review(self, text: str) -> ModerationDecision:
        """Classify ``text``."""
        ...


PROXIMITY_WINDOW: Final = 40
"""How many characters may separate two co-occurring terms."""


def _proximity(left: str, right: str, window: int = PROXIMITY_WINDOW) -> str:
    """Build a regex matching ``left`` and ``right`` within ``window`` characters.

    Order-independent, because a fraud script says both "share your OTP" and
    "OTP അയക്കൂ". Matching on ``.`` rather than ``\\W`` matters for Malayalam:
    the words between two terms are letters, so a non-word-character window
    never spans them.
    """
    return f"(?:(?:{left}).{{0,{window}}}(?:{right})|(?:{right}).{{0,{window}}}(?:{left}))"


_OTP = r"otp|ഒ\s*ടി\s*പി|ഒറ്റത്തവണ\s*പാസ്?.?വേഡ്"
_CREDENTIAL = r"password|പാസ്?.?വേഡ്|പിൻ\s*നമ്പർ|\bcvv\b|atm\s*pin|\bpin\b"
_SHARE = r"share|send|പറയ|അയക്ക|തരാമോ|തരൂ|നൽക|വേണം|ചൊല്ല"
_MONEY = r"പണം|രൂപ|money|transfer|അയക്ക|ഗൂഗിൾ\s*പേ|\bupi\b|\bgpay\b|₹"
_URGENT = r"അടിയന്തര|urgent|immediately|ഉടനെ|പെട്ടെന്ന്|ഇപ്പോൾ\s*തന്നെ"
_EMERGENCY = r"അപകട|ആശുപത്രി|അറസ്റ്റ്|hospital|accident|arrest|police\s*station|ജയില"
_AUTHORITY = r"മുഖ്യമന്ത്രി|മന്ത്രി|കമ്മീഷണർ|എസ്\s*ഐ|പോലീസ്|ഐ\s*പി\s*എസ്|collector"
_PRIZE = r"ലോട്ടറി|lottery|സമ്മാനം|prize|ജാക്ക്.?പോട്ട്"
_CLAIM = r"അടിച്ചു|\bwon\b|കിട്ടി|claim|ക്ലെയിം|ജയിച്ചു"

# (rule name, severity, pattern). Patterns are matched against the folded text.
FRAUD_PATTERNS: Final[tuple[tuple[str, Severity, str], ...]] = (
    ("otp_solicitation", Severity.BLOCK, _proximity(_OTP, _SHARE)),
    ("credential_solicitation", Severity.BLOCK, _proximity(_CREDENTIAL, _SHARE)),
    ("urgent_money_transfer", Severity.BLOCK, _proximity(_URGENT, _MONEY)),
    ("relative_in_trouble_scam", Severity.BLOCK, _proximity(_EMERGENCY, _MONEY, 60)),
    (
        "account_details_request",
        Severity.FLAG,
        r"അക്കൗണ്ട്\s*നമ്പർ|account\s*number|\bifsc\b|ബാങ്ക്\s*വിവര",
    ),
    (
        "impersonation_claim",
        Severity.FLAG,
        _proximity(r"ഞാൻ\s*(?:ആണ്|തന്നെ|ആണു)|this\s+is\s+the", _AUTHORITY, 30),
    ),
    ("prize_lottery_scam", Severity.FLAG, _proximity(_PRIZE, _CLAIM)),
)


def _normalize(text: str) -> str:
    """Fold text for matching: NFC, lower case, collapsed whitespace."""
    folded = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"\s+", " ", folded)


class PatternModerator:
    """Regex-based fraud-template detector.

    Args:
        patterns: Rules to apply. Defaults to :data:`FRAUD_PATTERNS`.
        extra_block_patterns: Deployment-specific rules that block.
    """

    def __init__(
        self,
        patterns: Iterable[tuple[str, Severity, str]] = FRAUD_PATTERNS,
        *,
        extra_block_patterns: Iterable[tuple[str, str]] = (),
    ) -> None:
        compiled: list[tuple[str, Severity, re.Pattern[str]]] = [
            (name, severity, re.compile(pattern, re.IGNORECASE))
            for name, severity, pattern in patterns
        ]
        compiled.extend(
            (name, Severity.BLOCK, re.compile(pattern, re.IGNORECASE))
            for name, pattern in extra_block_patterns
        )
        self._patterns = tuple(compiled)

    def review(self, text: str) -> ModerationDecision:
        """Classify ``text`` against every rule, taking the highest severity.

        Examples:
            >>> PatternModerator().review("നന്ദി, നാളെ കാണാം").severity.value
            'allow'
        """
        folded = _normalize(text)
        triggered: list[tuple[str, Severity]] = [
            (name, severity) for name, severity, pattern in self._patterns if pattern.search(folded)
        ]
        if not triggered:
            return ModerationDecision(Severity.ALLOW)
        severity = (
            Severity.BLOCK if any(s is Severity.BLOCK for _, s in triggered) else Severity.FLAG
        )
        return ModerationDecision(severity, tuple(name for name, _ in triggered))


class NameBlocklist:
    """Refuses voice enrolments labelled with a blocked name.

    Matching is on a folded form -- case, diacritics and non-alphanumerics
    removed -- and is substring-based, so "CM Pinarayi Vijayan (test)" matches an
    entry of "pinarayi vijayan". It is a speed bump against the obvious
    attempt, not identity verification; that is
    :mod:`mlvoice.voices.consent`'s job.

    Args:
        names: Blocked names.
    """

    def __init__(self, names: Iterable[str] = ()) -> None:
        self._folded = frozenset(filter(None, (self._fold(n) for n in names)))

    @classmethod
    def from_file(cls, path: str | Path) -> NameBlocklist:
        """Load a newline-delimited blocklist, ignoring blanks and ``#`` comments."""
        file = Path(path)
        if not file.is_file():
            # ``is_file`` rather than ``exists``: a misconfigured path can point
            # at a directory, and reading one raises IsADirectoryError.
            log.warning(
                "blocklist file not found or not a file; enrolment name checks disabled",
                path=str(file),
            )
            return cls()
        lines = (line.strip() for line in file.read_text(encoding="utf-8").splitlines())
        return cls(line for line in lines if line and not line.startswith("#"))

    @staticmethod
    def _fold(name: str) -> str:
        decomposed = unicodedata.normalize("NFKD", name.lower())
        return "".join(ch for ch in decomposed if ch.isalnum())

    def __len__(self) -> int:
        """Number of blocked names."""
        return len(self._folded)

    def is_blocked(self, name: str) -> bool:
        """True if ``name`` contains, or is contained by, a blocked entry."""
        folded = self._fold(name)
        if not folded:
            return False
        return any(entry in folded or folded in entry for entry in self._folded)
