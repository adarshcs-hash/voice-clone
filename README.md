# mlvoice — Malayalam text-to-speech and voice cloning

A production-shaped platform for synthesising Malayalam speech and cloning
Malayalam voices, built around the parts that actually decide quality for this
language: a real text frontend, a measured evaluation harness, and consent and
provenance enforced in code rather than in a policy document.

```bash
make install-dev
make check                      # lint, types, tests
mlvoice eval frontend           # score the Malayalam text frontend, no GPU needed
mlvoice serve                   # http://127.0.0.1:8000/docs
```

## Why this shape

Off-the-shelf multilingual TTS lists Malayalam among dozens of languages and
gets it *nearly* right, which is the worst outcome: intelligible enough to ship,
wrong in ways a native speaker hears in the first sentence. The errors are
systematic, not random, and almost all of them happen **before** the acoustic
model:

| Problem | What goes wrong | Where it is handled |
|---|---|---|
| Samvruthokaram | `നാട്` read as `naːʈ` or `naːʈu` instead of `naːʈɨ` | [`text/g2p.py`](src/mlvoice/text/g2p.py) |
| Legacy encodings | `എൻറെ` and `എന്റെ` treated as different words | [`text/unicode_norm.py`](src/mlvoice/text/unicode_norm.py) |
| Gemination | `പത്ത്` and a single `ത` collapsed together | [`text/g2p.py`](src/mlvoice/text/g2p.py) |
| Nasal clusters | `ണ്ട` read unvoiced, `ം` not assimilating | [`text/g2p.py`](src/mlvoice/text/g2p.py) |
| Numbers | `1,25,000` read as thousands, or digit by digit | [`text/numbers.py`](src/mlvoice/text/numbers.py) |
| Code-mixing | English mid-sentence read as Malayalam letters, or dropped | [`text/codemix.py`](src/mlvoice/text/codemix.py) |
| Manglish | Latin-script Malayalam read as English | [`text/translit.py`](src/mlvoice/text/translit.py) |
| Agglutination | 38-character words split mid-morpheme | [`text/chunker.py`](src/mlvoice/text/chunker.py) |

So the text frontend is the largest and best-tested component here, it has no
dependency on any model runtime, and it is gated in CI on every commit.

## Start by measuring, not by training

`mlvoice eval frontend` runs the [Malayalam hard test
set](src/mlvoice/eval/testset.py) — 75 sentences, 18 categories, each chosen
because it probes one specific failure — through the text frontend and asserts
the output. It takes milliseconds and needs no weights.

Before choosing or training anything, run the same test set through every
candidate system, including the commercial APIs, and score the audio with
native speakers using `mlvoice eval testset --instructions`. That measurement
tells you where the real gap is. Everything else in this repository is
downstream of it.

## Architecture

```
mlvoice/
├── text/        Malayalam text frontend — no model dependency
│   ├── chars.py         Character inventory and classification
│   ├── unicode_norm.py  Chillu, legacy nta, archaic signs, digits
│   ├── numbers.py       Cardinals, ordinals, decimals, lakh/crore
│   ├── expand.py        Dates, times, currency, units, abbreviations
│   ├── translit.py      Manglish → Malayalam script
│   ├── codemix.py       Script segmentation and routing
│   ├── g2p.py           Phonemiser with the Malayalam rule cascade
│   ├── chunker.py       Sentence and clause chunking with pause strengths
│   └── pipeline.py      The frontend, assembled
├── audio/       IO, loudness (BS.1770), quality gates, VAD, denoising
├── data/        Corpus manifests and the preparation pipeline
├── tts/         Synthesiser contract, IndicF5 backend, deterministic stub
├── voices/      Enrolment, consent challenge/verification, persistence
├── safety/      Watermarking, moderation, name blocklist
├── eval/        Metrics, the hard test set, the evaluation harness
├── api/         FastAPI service
├── synthesis.py The application service that composes all of the above
└── cli.py       Operational CLI
```

The dependency graph is acyclic and layered: `text` and `audio` know nothing
about models; `tts` knows nothing about HTTP; shared capabilities (ASR, speaker
verification, MOS estimation) are protocols in
[`protocols.py`](src/mlvoice/protocols.py) so `voices` and `eval` do not depend
on each other.

## The model

The default target is
[`ai4bharat/IndicF5`](https://huggingface.co/ai4bharat/IndicF5) — an F5-TTS
flow-matching model trained on 1,417 hours across 11 Indian languages including
Malayalam, which clones zero-shot from a reference clip plus that clip's
transcript. It is the strongest open starting point for Malayalam, and
fine-tuning it on Malayalam-only data is a far better use of a small budget than
training from scratch.

Backends are registered in [`tts/registry.py`](src/mlvoice/tts/registry.py) and
selected by configuration:

```bash
pip install -e ".[indicf5]"
MLVOICE_TTS_BACKEND=indicf5 MLVOICE_MODEL_REVISION=<commit-sha> mlvoice serve
```

The `indicf5` extra is separate from `models` because the model's bundled code
pulls in a large dependency tree of its own — those are the model's
requirements, not this project's, and they have no place in a deployment that
only needs the inference path.

Two things in that extra are load-bearing and easy to get wrong:

- **`f5_tts` must be AI4Bharat's fork**, installed from GitHub, not the
  `f5-tts` package on PyPI. Both expose the same import name; only the fork has
  the API IndicF5's code calls.
- **`transformers` must be `<4.50`.** 4.51.0 made meta-device initialisation
  unconditional, under which the model cannot load; AI4Bharat's own
  requirements say `<4.50`.

The `dummy` backend is a deterministic phoneme-driven signal generator. It is
not a model; it exists so the API contract, streaming, watermarking and the
whole test suite run on a CPU-only box in seconds. Production configuration
refuses it.

> **`ai4bharat/IndicF5` is a gated repository.** Request access on its model
> page, then authenticate (`hf auth login`, or set `HF_TOKEN`) for the account
> that was granted access. Without that, loading fails with a 401
> `GatedRepoError`.
>
> It also requires `trust_remote_code=True`, which is remote code execution by
> design. `MLVOICE_MODEL_REVISION` must pin a commit in production; startup
> refuses an unpinned revision. Pin `transformers<5`: the model's bundled code
> targets the 4.x API.

## Voice cloning requires verified consent

Enrolment is a two-step spoken challenge, because a one-step flow cannot
establish consent — any audio could be supplied for both roles.

```bash
# 1. Issue a challenge: the subject's name, today's date, and a random nonce.
curl -sX POST localhost:8000/v1/voices/challenge \
  -H 'X-API-Key: dev-local-key' -H 'content-type: application/json' \
  -d '{"subject_name":"രാജൻ നായർ"}'

# 2. Enrol with the reference clip and a recording of the subject reading it.
curl -sX POST localhost:8000/v1/voices \
  -H 'X-API-Key: dev-local-key' \
  -F name='Rajan' -F reference_text='ഇത് എന്റെ ശബ്ദ സാമ്പിൾ ആണ്' \
  -F consent_token="$TOKEN" \
  -F reference_audio=@reference.wav -F consent_audio=@consent.wav
```

The service verifies that the phrase was spoken (ASR, gated on CER) **and** that
it is the same speaker as the reference clip (speaker verification, gated on
cosine similarity). The nonce is what binds the recording to this enrolment at
this time; without it, a recording of someone saying "I consent" could be lifted
from anywhere.

Consent is re-checked on **every synthesis request**, not only at enrolment.
Revoking a consent record disables every voice that relied on it, and deleting a
voice deletes the reference audio from disk.

This is not identity verification — it proves the same person recorded both
clips, not who that person is. See
[`docs/safety-and-compliance.md`](docs/safety-and-compliance.md).

## Provenance

Every clip returned by `POST /v1/tts` carries a watermark whose payload is a key
into the request log, and `POST /v1/watermark/detect` answers "did we generate
this?".

Two implementations: **AudioSeal** (robust to compression, resampling, cropping
— what production should run) and a **band-limited spread-spectrum** watermark
implemented in NumPy, which survives gain changes and lossless format
conversion but *not* lossy compression. The default install uses the latter so
that a stock deployment still watermarks; the limitation is documented in
[`safety/watermark.py`](src/mlvoice/safety/watermark.py) rather than hidden.

Streamed audio (`POST /v1/tts/stream`) is **not** watermarked — the payload needs
the whole utterance — and the response says so in `X-Watermarked: false`.

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/tts` | Synthesise; returns a watermarked WAV |
| `POST /v1/tts/stream` | Chunked WAV stream, low time-to-first-byte |
| `POST /v1/text/analyze` | Every frontend stage — answers "why did it say that?" |
| `POST /v1/transcribe` | Transcribe a clip — used to fill in the reference transcript |
| `POST /v1/voices/challenge` | Issue a consent challenge |
| `POST /v1/voices` | Enrol a voice |
| `GET /v1/voices`, `GET /v1/voices/{id}` | List and read voices |
| `POST /v1/voices/{id}/disable`, `DELETE /v1/voices/{id}` | Disable, delete |
| `POST /v1/consents/{id}/revoke` | Revoke consent (cascades) |
| `POST /v1/watermark/detect` | Provenance check |
| `GET /healthz`, `GET /readyz`, `GET /v1/info` | Liveness, readiness, capabilities |
| `GET /ui` | Browser client |

`/readyz` returns 503 until the model has loaded, so a rolling deploy never
routes into a cold replica. Every response carries `X-Request-Id`, and every log
line emitted while handling that request carries the same id.

## Web client

`GET /ui` serves a single page — no build step, no CDN, no framework, just an
HTML file and a script the API process itself hands out. Upload or record a
voice, record the consent sentence, and type what it should say.

Two decisions shape it:

**The reference transcript is transcribed, not typed.** Cloning needs to know
what the reference clip *says*, and asking a user to type that out is both a bad
first impression and the source of the worst failure this system has: type the
text you want *generated* instead of the words in the clip, and you get a
perfect clone of the voice saying something between the two. So the page uploads
the clip to `POST /v1/transcribe`, fills the field in, and asks only that you
correct it — with the field labelled "what the reference clip says" to make the
distinction hard to miss. If transcription is unavailable the field stays
editable and the page says so, rather than failing.

**The browser converts audio, not the server.** Every clip goes through
`AudioContext.decodeAudioData` and comes out 16-bit PCM WAV before it is
uploaded. One code path covers a phone's `.m4a`, an `.mp3`, and the WebM/Opus a
`MediaRecorder` produces, and the server needs no codec beyond libsndfile —
server-side conversion would mean shipping ffmpeg in the image.

The page also exposes `POST /v1/text/analyze`, so you can see the chunks and
phonemes the model will actually be given before paying for a generation.

## Corpus preparation

```bash
mlvoice data prepare corpus.tsv \
  --output-dir data/processed --manifest data/manifest.jsonl \
  --source imasc --license-id CC-BY-4.0 --commercial
mlvoice data stats data/manifest.jsonl
```

Per utterance: resample → trim silence → denoise → loudness-normalise → quality
gate → text frontend → ASR verification → manifest row.

Two decisions matter more than the rest:

- **ASR verification is the real filter.** Source transcripts are wrong often
  enough to matter. Re-transcribing and rejecting rows whose CER against the
  transcript is too high is worth more to final quality than any
  hyperparameter. Wire a [`Transcriber`](src/mlvoice/protocols.py) in to enable
  it; without one the manifest records the corpus as unverified.
- **Splits are speaker-disjoint.** A validation set sharing speakers with train
  measures memorisation, which for a cloning model is exactly the quantity being
  faked. Assignment hashes the speaker id, so it is stable as the corpus grows.

Malayalam data sources, sizes and licence terms — including which ones forbid
commercial use — are in
[`docs/data-and-licensing.md`](docs/data-and-licensing.md). The manifest carries
`license_id` and `commercial_use_permitted` on every row so a non-commercial
dataset cannot silently end up in a commercial training run.

## Configuration

Environment variables, `MLVOICE_` prefixed; see [`.env.example`](.env.example)
and [`config.py`](src/mlvoice/config.py). Settings are validated once at
startup, and `MLVOICE_ENV=production` **refuses to start** unless:

- `MLVOICE_API_KEYS` is set,
- `MLVOICE_REQUIRE_CONSENT` and `MLVOICE_WATERMARK_ENABLED` are on,
- `MLVOICE_CONSENT_SIGNING_KEY` and `MLVOICE_WATERMARK_KEY` are set,
- `MLVOICE_MODEL_REVISION` pins the weights,
- and the backend is not `dummy`.

Failing at startup is the point: these are the invariants that make the system
safe to run, and a deployment that quietly disabled one would look identical
from the outside.

## Development

```bash
make install-dev
make lint typecheck test     # or: make check
make cov
make serve
make docker
```

`ruff` and `mypy --strict` are clean; the fast test suite needs no model weights
and no network. Tests that require weights are marked `slow` and excluded by
default. `make test-browser` drives the web client in a real Chromium — those
tests are marked `browser` and skip themselves when no browser is installed,
so they never fail a contributor who does not want the download. Current state:
**653 tests, 94% branch coverage**.

## What is honest about this

- The **text frontend** is complete and tested. Its linguistic tables are the
  project's ground truth and are documented as needing native-speaker review
  before production — see [`docs/malayalam-linguistics.md`](docs/malayalam-linguistics.md).
- **Manglish transliteration** is rule-based and therefore approximate; Manglish
  is not a standardised orthography, and single letters are genuinely ambiguous.
  A trained sequence-to-sequence transliterator is the production answer, and
  `transliterate()` is the interface it slots into.
- **English words inside Malayalam** are passed through by default rather than
  transliterated. There is no honest rule-based mapping from English spelling to
  Malayalam script, so the module offers a loanword lexicon and a documented
  seam instead of a bad guess.
- **Intervocalic voicing** is implemented but off by default: it is dialect- and
  register-dependent, and a neural acoustic model trained on real speech learns
  that variation better than a rule can impose it.
- **The rate limiter is per process.** With several replicas the effective limit
  is `replicas × rate`. It is a safety valve, not a quota; a quota needs shared
  state.
- **No model weights are trained here.** This is the platform: the frontend, the
  data pipeline, the safety layer, the evaluation harness and the serving stack
  that a fine-tune plugs into. [`docs/roadmap.md`](docs/roadmap.md) sets out the
  fine-tuning path, with realistic time and cost.

## Licence

Apache 2.0 — see [LICENSE](LICENSE). That covers this source code only. Model
weights and speech corpora carry their own terms, some of them
non-commercial. Read [`docs/data-and-licensing.md`](docs/data-and-licensing.md)
before shipping.
