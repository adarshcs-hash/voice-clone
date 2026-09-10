# Measuring first

The first task on this project is not training. It is building the instrument
that tells you whether training is needed, and where.

## Step 1: run the frontend gate (minutes)

```bash
mlvoice eval frontend
```

75 sentences, 18 categories, each chosen because it probes one specific way a
Malayalam voice goes wrong. No model, no GPU, no network. Ten of the cases carry
an asserted normalised output, so this runs in CI as a build gate rather than a
report.

```bash
mlvoice eval frontend -o report.json --fail-under 1.0
mlvoice eval testset                       # coverage per category
mlvoice eval testset --category numbers    # the cases themselves
```

## Step 2: benchmark every candidate system (a week)

Run the same 75 sentences through every system you might use — including the
commercial APIs — and score the audio with native speakers.

```bash
mlvoice eval testset --category samvruthokaram   # tab-separated: id, text, probe
mlvoice eval testset --instructions              # the rating instructions
```

Systems worth including: ElevenLabs, `ai4bharat/IndicF5`,
`ai4bharat/indic-parler-tts`, `facebook/mms-tts-mal`, Google and Azure
Malayalam voices, and any IMaSC-derived checkpoint.

**Do not skip the human scoring.** Objective metrics cannot see the errors that
matter most here — an ASR model that shares the TTS model's confusions will
happily transcribe a mispronounced word back to the right characters, and CER is
blind to prosody entirely.

Four axes, rated 1–5 independently, at least five raters per clip, reported as
median with the inter-rater range:

1. **Naturalness** — does it sound like a person speaking Malayalam?
2. **Pronunciation** — with samvruthokaram, consonant length and nasal clusters
   marked specifically.
3. **Intelligibility** — understood first time, without reading the text?
4. **Speaker similarity** — for cloned voices, against the reference clip.

The category breakdown is the deliverable. A model that passes `numbers` and
fails `samvruthokaram` implies completely different work than the reverse, and a
single average hides exactly that.

## Step 3: objective metrics between rating rounds

```bash
mlvoice eval synthesis -o synth.json
```

Reports real-time factor and objective audio quality out of the box. Add ASR and
speaker verification to get the rest:

```python
from mlvoice.eval import EvaluationHarness, EcapaSpeakerVerifier
from mlvoice.protocols import Transcriber

class IndicConformer(Transcriber):
    def transcribe(self, audio): ...

harness = EvaluationHarness(
    transcriber=IndicConformer(),
    speaker_verifier=EcapaSpeakerVerifier(),
)
report = harness.evaluate_synthesis(synthesizer, prompt=prompt)
print(report.to_json())
```

### Metric choices, and why

| Metric | Notes |
|---|---|
| **CER** | The primary objective gate. Malayalam is agglutinative — one wrong morpheme fails an entire long word — so WER is coarse and unstable across systems that segment differently. Scoring normalisation applies Malayalam Unicode normalisation first, so a chillu spelling difference is not counted as an error |
| WER | Reported for comparability with published numbers; never the primary signal |
| Speaker similarity | Cosine similarity of ECAPA embeddings. **Calibrate the threshold on your own data** — scores are not comparable across models |
| UTMOS / DNSMOS | No-reference perceptual quality, via the `MosEstimator` protocol |
| Real-time factor | The number that decides whether a deployment is affordable |

Metrics are implemented dependency-free; the test suite cross-checks them
against `jiwer` when it is installed rather than depending on it, so CI on a
bare box still measures the thing it gates on.

Extras: `pip install "mlvoice[eval]"` adds `jiwer` for that cross-check;
`pip install "mlvoice[speaker]"` adds the ECAPA speaker verifier, which needs
torch.

## Step 4: extend the test set

The shipped set is a **seed of 75 cases**. A production gate wants 300 or more,
with at least five per category. Extend `eval/testset.py` in place — the ids are
stable and referenced by report history, so add new ones rather than
renumbering.

Priorities when extending:

1. More place and person names — the highest-frequency real-world failure.
2. More code-mix, at realistic density (English in most sentences, not one word
   in twenty).
3. Domain sentences for whatever you are actually shipping: news bulletins,
   audiobook prose, IVR prompts, product descriptions.
4. Dialect-marked sentences, if you intend to support more than the standard
   formal register.
5. Adversarial input: mixed encodings, stray Latin, emoji, very long
   agglutinated words, numbers in unusual formats.

Add `expected_text` wherever the frontend has exactly one correct output. Those
cases become automatic gates and stop being a human-review burden.
