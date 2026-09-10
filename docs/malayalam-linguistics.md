# The Malayalam rules this project encodes

Every linguistic claim the code makes lives in a table, and every table is
listed here with the module that owns it. Changing one of these is changing a
claim about Malayalam, and should be reviewed by a native speaker rather than by
a Python reviewer.

## 1. Script and encoding — `text/unicode_norm.py`

Malayalam admits several byte sequences for the same word. Unnormalised, a TTS
tokenizer sees them as different tokens, which splits training signal and
produces mispronunciation at inference.

| Variation | Example | Handling |
|---|---|---|
| Atomic vs ZWJ chillu | `ൻ` vs `ന` + `്` + ZWJ | Compose to the atomic chillu |
| Legacy `nta` | `ൻ` + `റ` for `ന്റ` | Repair to `ന` + `്` + `റ` |
| AU sign | `ൌ` (U+0D4C) vs `ൗ` (U+0D57) | Map to `ൗ` |
| Dot reph | `ൎ` (U+0D4E) | Map to `ർ` |
| Archaic letters | `ൟ`, `ൔ`, `ൕ`, `ൖ` | Map to modern equivalents |
| Zero-width controls | ZWNJ, stray ZWJ | Strip (no phonetic content) |
| Malayalam digits | `൧൨൩`, `൰൱൲` | Convert to ASCII digits |

**The legacy `nta` repair is the one that matters most in practice.** Pre-Unicode
converters and older keyboards emit `ൻ` + `റ` where the correct sequence is
`ന്റ`; it is the most common defect in real Malayalam corpora, and until it is
repaired, `എൻറെ` and `എന്റെ` are different words to the model.

### The one thing deliberately not done by default

`ൻ` followed by `ന` or `മ`, and `ൺ` followed by `ട` or `മ`, are *usually*
legacy mis-encodings — but not always: `മുൻനിര` ("front row") is a legitimate
compound with a real chillu before `ന`. Repairing these is available behind
`aggressive_legacy=True`, which is safe for a single corpus known to be
legacy-encoded and unsafe as a default for user input.

## 2. Numerals — `text/numbers.py`

Malayalam numerals are not a positional composition of digit names.

**Tens plus units take an oblique stem and a `യ` glide before a vowel-initial
unit:**

```
ഇരുപത് (20) + അഞ്ച് (5)  → ഇരുപത്തിയഞ്ച്     glide inserted
ഇരുപത് (20) + രണ്ട് (2)  → ഇരുപത്തിരണ്ട്      consonant-initial, no glide
```

**Hundreds are lexicalised**, not `n` + "hundred":

| | | | | |
|---|---|---|---|---|
| 100 `നൂറ്` | 200 `ഇരുനൂറ്` | 300 `മുന്നൂറ്` | 400 `നാനൂറ്` | 500 `അഞ്ഞൂറ്` |
| 600 `അറുനൂറ്` | 700 `എഴുനൂറ്` | 800 `എണ്ണൂറ്` | 900 `തൊള്ളായിരം` | |

**Thousand multipliers are irregular:**

| | | | | |
|---|---|---|---|---|
| 1000 `ആയിരം` | 2000 `രണ്ടായിരം` | 3000 `മൂവായിരം` | 4000 `നാലായിരം` | 5000 `അയ്യായിരം` |
| 6000 `ആറായിരം` | 7000 `ഏഴായിരം` | 8000 `എണ്ണായിരം` | 9000 `ഒൻപതിനായിരം` | 10000 `പതിനായിരം` |

**Grouping is lakh/crore**, so 10⁶ is `പത്ത് ലക്ഷം` (ten lakh), not "one
million". The multiplier "one" is attributive `ഒരു`, never `ഒന്ന്`:
100,000 is `ഒരു ലക്ഷം`.

### Known simplification

Gemination sandhi across a magnitude boundary is not applied. 999 comes out as
`തൊള്ളായിരത്തിതൊണ്ണൂറ്റിയൊൻപത്` where careful orthography geminates to
`തൊള്ളായിരത്തിത്തൊണ്ണൂറ്റിയൊൻപത്`. The two are homophonous once the phonemiser
applies its own gemination rules, and skipping the sandhi keeps the tables
auditable.

Coverage: every value 0–99 is asserted individually; larger values by golden
cases. See `tests/unit/test_text_numbers.py`.

## 3. Pronunciation — `text/g2p.py`

Four systematic gaps between Malayalam orthography and pronunciation account for
most of what a listener hears as "wrong" from a multilingual model.

### Samvruthokaram (the half-`u`) — the most audible error

A word-final chandrakkala, and a word-final `ു`, are both realised as a short
central vowel [ɨ]:

```
നാട്  → [naːɖɨ]      not [naːʈ] and not [naːʈu]
അതു   → [ad̪ɨ]        not [ad̪u]
```

### Gemination

A doubled consonant is phonemically long, and the contrast is meaningful:
`പത്ത്` is [pat̪ːɨ], distinct from a single [t̪].

### Nasal assimilation and post-nasal voicing

The anusvara `ം` takes the place of articulation of a following stop, and a
nasal followed by a voiceless stop voices it:

| Cluster | Realisation |
|---|---|
| `ങ്ക` | [ŋɡ] |
| `ഞ്ച` | [ɲdʒ] |
| `ണ്ട` | [ɳɖ] |
| `ന്ത` | [nd̪] |
| `മ്പ` | [mb] |
| `ം` + `ഗ` | [ŋɡ] |

### Cluster idiosyncrasies

`റ്റ` is a geminate alveolar stop [tː], not a doubled trill. `ന്റ` is [nd̪], not
[nr].

### Intervocalic voicing — implemented, off by default

Lenition of a single intervocalic voiceless stop (`ക` → [ɡ], `ത` → [d̪]) is real
Malayalam allophony, but it is dialect- and register-dependent. It is available
behind `G2PConfig(intervocalic_voicing=True)` and **off by default**: a neural
acoustic model trained on real speech learns that variation from data, and
forcing one dialect's realisation into the phoneme string removes the model's
ability to.

### Symbol set

Phonemes are emitted as IPA and as a stable ASCII identifier
(`PHONEME_TABLE`). Model vocabularies should be built from the ASCII form —
tokenizers handle combining diacritics inconsistently, and a stable symbol set
makes a vocabulary reproducible. A test asserts the two tables cannot drift
apart.

## 4. Transliteration — `text/translit.py`

Manglish is not a standardised orthography and single Latin letters are
genuinely ambiguous: `t` may be `ത`, `ട` or `റ്റ`; `o` may be `ഒ` or `ഓ`. A rule
system cannot be exact.

The implementation is a documented longest-match scheme following Mozhi
conventions where they are unambiguous, plus a correction lexicon for
high-frequency words. Detection requires **positive evidence** by default — a
Malayalam consonant digraph (`zh`, `nj`, `ng`, a geminate), a Malayalam
inflectional ending, or lexicon membership — because an English noun with no
such signal is better left alone than transliterated into gibberish. Callers
who know their input is Manglish can set
`ManglishDetection.PERMISSIVE`.

The production answer is a trained sequence-to-sequence transliterator;
`transliterate()` is the interface it slots into.

## 5. English inside Malayalam — `text/codemix.py`

English spelling is not phonemic, so there is no honest rule-based mapping from
English orthography to Malayalam script. The default policy therefore passes
English through unchanged. Two escape hatches exist:

- **`loanword_lexicon`** — pin the domain vocabulary that must be pronounced
  correctly. This is the practical answer for a specific deployment.
- **`LatinPolicy.TRANSLITERATE`** — the seam where a measured English
  grapheme-to-phoneme model is dropped in.

Acronyms are spelled with Malayalam letter names (`BJP` → `ബി ജെ പി`), which is
how they are actually read aloud.

## 6. Chunking — `text/chunker.py`

Malayalam is agglutinative: single words routinely exceed 30 characters
(`അന്താരാഷ്ട്രവിമാനത്താവളത്തിലേക്കുള്ളവഴിയിലൂടെപോകുന്നവർ`). Splitting inside one
produces two mispronounced fragments, so the chunker **never** splits a word: it
tries sentence boundaries, then clause boundaries, then word boundaries, and
emits an over-long word whole.

## What needs native-speaker review before production

1. The numeral tables above 100,000, especially the lakh/crore stem forms.
2. The month names and the date word order in `text/expand.py`.
3. The Latin letter names used for acronyms.
4. Whether `intervocalic_voicing` should be enabled for your target dialect.
5. The `ഒൻപത്` / `ഒമ്പത്` choice for 9 — both are current; this project uses the
   former consistently.
6. The Manglish correction lexicon, which is inevitably a matter of judgement.

Run `mlvoice eval frontend` after any change: the hard test set asserts the
outputs that are settled.
