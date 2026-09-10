"""Malayalam character inventory and classification.

Every other module in :mod:`mlvoice.text` builds on the constants here rather
than on inline literals, so the inventory can be audited in one place. Names
follow the Unicode character names for the Malayalam block (U+0D00-U+0D7F).
"""

from __future__ import annotations

# --- Format controls --------------------------------------------------------
ZWNJ = "‌"
"""Zero-width non-joiner: forces a visible chandrakkala instead of a ligature."""
ZWJ = "‍"
"""Zero-width joiner: after a virama it selects the chillu form."""

# --- Signs -----------------------------------------------------------------
CANDRABINDU = "ഁ"
ANUSVARA = "ം"  # ം - word-final /m/, or a homorganic nasal before a stop
VISARGA = "ഃ"  # ഃ
VIRAMA = "്"  # ് - chandrakkala; deletes the inherent /a/
DOT_REPH = "ൎ"  # ൎ - historic superscript /r/, logically pre-consonantal
AU_LENGTH_MARK = "ൗ"  # ൗ - modern AU sign
VOWEL_SIGN_AU = "ൌ"  # ൌ - legacy AU sign, visually equivalent

# --- Independent vowels ----------------------------------------------------
INDEPENDENT_VOWELS = "അആഇഈഉഊഋഌഎഏഐഒഓഔ"
"""അ ആ ഇ ഈ ഉ ഊ ഋ ഌ എ ഏ ഐ ഒ ഓ ഔ"""

ARCHAIC_II = "ൟ"  # ൟ

# --- Dependent vowel signs (matras) ----------------------------------------
VOWEL_SIGNS = (
    "ാിീുൂൃൄ"  # ാ ി ീ ു ൂ ൃ ൄ
    "െേൈൊോൌ"  # െ േ ൈ ൊ ോ ൌ
    "ൗ"  # ൗ
    "ൢൣ"  # ൢ ൣ
)

# --- Consonants ------------------------------------------------------------
CONSONANTS = (
    "കഖഗഘങ"  # ക ഖ ഗ ഘ ങ
    "ചഛജഝഞ"  # ച ഛ ജ ഝ ഞ
    "ടഠഡഢണ"  # ട ഠ ഡ ഢ ണ
    "തഥദധന"  # ത ഥ ദ ധ ന
    "ഩ"  # ഩ (archaic tta-na)
    "പഫബഭമ"  # പ ഫ ബ ഭ മ
    "യരറലളഴവ"  # യ ര റ ല ള ഴ വ
    "ശഷസഹ"  # ശ ഷ സ ഹ
    "ഺ"  # ഺ (archaic ttta)
)

# --- Chillu (pure consonant) letters ---------------------------------------
CHILLU_M = "ൔ"  # ൔ
CHILLU_Y = "ൕ"  # ൕ
CHILLU_LLL = "ൖ"  # ൖ
CHILLU_NN = "ൺ"  # ൺ
CHILLU_N = "ൻ"  # ൻ
CHILLU_RR = "ർ"  # ർ
CHILLU_L = "ൽ"  # ൽ
CHILLU_LL = "ൾ"  # ൾ
CHILLU_K = "ൿ"  # ൿ

CHILLUS = CHILLU_NN + CHILLU_N + CHILLU_RR + CHILLU_L + CHILLU_LL + CHILLU_K
ARCHAIC_CHILLUS = CHILLU_M + CHILLU_Y + CHILLU_LLL

CHILLU_TO_BASE: dict[str, str] = {
    CHILLU_NN: "ണ",  # ൺ -> ണ
    CHILLU_N: "ന",  # ൻ -> ന
    CHILLU_RR: "ര",  # ർ -> ര
    CHILLU_L: "ല",  # ൽ -> ല
    CHILLU_LL: "ള",  # ൾ -> ള
    CHILLU_K: "ക",  # ൿ -> ക
}
BASE_TO_CHILLU: dict[str, str] = {base: chillu for chillu, base in CHILLU_TO_BASE.items()}

# --- Digits ----------------------------------------------------------------
MALAYALAM_DIGITS: dict[str, str] = {
    "൦": "0",
    "൧": "1",
    "൨": "2",
    "൩": "3",
    "൪": "4",
    "൫": "5",
    "൬": "6",
    "൭": "7",
    "൮": "8",
    "൯": "9",
}
MALAYALAM_NUMBER_SIGNS: dict[str, str] = {
    "൰": "10",
    "൱": "100",
    "൲": "1000",
}

# --- Aggregate sets used for classification --------------------------------
MALAYALAM_BLOCK = frozenset(chr(cp) for cp in range(0x0D00, 0x0D80))
CONSONANT_SET = frozenset(CONSONANTS)
CHILLU_SET = frozenset(CHILLUS + ARCHAIC_CHILLUS)
VOWEL_SIGN_SET = frozenset(VOWEL_SIGNS)
INDEPENDENT_VOWEL_SET = frozenset(INDEPENDENT_VOWELS + ARCHAIC_II)
SIGN_SET = frozenset({CANDRABINDU, ANUSVARA, VISARGA})


def is_malayalam(ch: str) -> bool:
    """True if ``ch`` lies in the Malayalam Unicode block."""
    return ch in MALAYALAM_BLOCK


def is_consonant(ch: str) -> bool:
    """True for a base consonant letter (not a chillu)."""
    return ch in CONSONANT_SET


def is_chillu(ch: str) -> bool:
    """True for a chillu (pure-consonant) letter."""
    return ch in CHILLU_SET


def is_vowel_sign(ch: str) -> bool:
    """True for a dependent vowel sign (matra)."""
    return ch in VOWEL_SIGN_SET


def is_independent_vowel(ch: str) -> bool:
    """True for an independent vowel letter."""
    return ch in INDEPENDENT_VOWEL_SET


def has_malayalam(text: str) -> bool:
    """True if ``text`` contains at least one Malayalam character."""
    return any(is_malayalam(ch) for ch in text)
