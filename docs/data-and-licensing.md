# Malayalam speech data: sources, sizes and licence terms

**Read this before training.** The Apache 2.0 licence on this source code does
not extend to model weights or speech corpora. Several of the datasets below
forbid commercial use, and at least one widely used TTS model is
non-commercial. Getting this wrong is not a technical problem you can fix later.

Every manifest row carries `license_id` and `commercial_use_permitted`
(`data/manifest.py`), so a training run filters on provenance rather than on
someone's memory of where a directory came from.

## Open Malayalam speech corpora

| Corpus | Malayalam size | Speakers | Character | Notes |
|---|---|---|---|---|
| **IndicVoices-R** | ~82.6 h (5.5 h read + 77.1 h extempore) | 462 | Natural, conversational | Restored for TTS from an ASR corpus; the largest open Malayalam TTS-grade set |
| **IMaSC** (ICFOSS) | ~50 h, 34,473 pairs | 8 | Studio, read | Highest per-speaker consistency available openly |
| **IndicTTS** (IIT Madras) | ~10 h male + ~10 h female | 2 | Studio, read | Very clean; only two voices |
| **Rasa / LIMMITS** | included in IndicF5's 1,417 h | — | Expressive / multi-speaker | Already inside the base model |
| **Shrutilipi** | hundreds of hours (All India Radio) | many | Broadcast news, noisier | ASR-grade; useful for pretraining, not final fine-tuning |
| **Common Voice (ml)** | small | many | Crowd-sourced, variable | Best used for evaluation |
| **OpenSLR Malayalam** | small | few | Crowd-sourced | Useful for evaluation and augmentation |

Realistically usable open data: **roughly 150–250 hours**. That is enough for a
strong fine-tune of an existing multilingual model. It is *not* enough to train
a foundation model from scratch, and attempting that with this much data is the
most common way this kind of project fails.

Verify the current size and terms of each corpus at its own source before
building on it. Dataset licences change, and re-releases change the numbers.

## Licences: check these individually

- **Corpus licences vary between the datasets above, and some are
  research-only.** Each must be checked at source, per dataset, and recorded in
  `license_id` on every row it contributes.
- **`ai4bharat/IndicF5`** — check the model card's licence before commercial
  deployment.
- **Coqui XTTS-v2** — the Coqui Public Model Licence is **non-commercial**. It
  also has no Malayalam support, so it is not a candidate here regardless.
- **Broadcast and streaming audio** (news channels, YouTube, film) — do not
  scrape. Kerala's media output is attractive training data and almost all of it
  is someone's copyright, often with a performer's separate rights on top.

## Your own recordings are the actual asset

Open data gets you a working system. It does not get you a *better* system than
everyone else building on the same open data.

A studio corpus of **20–40 hours across 10–20 speakers**, deliberately balanced,
is the thing nobody else has:

- **Dialect spread** — Thiruvananthapuram, Kottayam, Thrissur, Kozhikode,
  Malabar, Kasaragod. Open corpora skew heavily to the formal read register of
  broadcasting, which is why models trained on them sound like a news anchor
  reading a WhatsApp message. `Dialect` on the manifest is what makes this skew
  visible; without the field, a corpus looks balanced because nobody measured.
- **Register spread** — read, conversational, expressive.
- **Phonetic coverage** — scripts written to cover every conjunct, every chillu,
  samvruthokaram in each position, and code-mixed sentences.
- **Consent from the outset** — record the consent challenge
  (`voices/consent.py`) at the same session. Retrofitting consent to a corpus is
  usually impossible.

Indicative cost in Kerala: **₹3–8 lakh** for 20–40 hours including studio time,
speaker fees and transcription review.

## Preparing what you have

```bash
mlvoice data prepare corpus.tsv \
  --output-dir data/processed --manifest data/manifest.jsonl \
  --source imasc --license-id CC-BY-4.0 --commercial
mlvoice data stats data/manifest.jsonl
```

`data stats` reports hours, speakers, and the distribution by dialect, gender,
source and commercial-use flag. Read it before every training run — it is the
cheapest way to notice that 80% of your "balanced" corpus is two speakers from
one district.

### ASR verification is the step worth the most

Source transcripts are wrong often enough to matter: misaligned segments,
truncated sentences, the wrong take. Re-transcribing each clip and rejecting
rows whose CER against the transcript exceeds a threshold removes those, and it
improves final quality more than any hyperparameter you will tune afterwards.

Use an Indic-specific recogniser (IndicConformer, IndicWhisper). General
multilingual Whisper is materially weaker on Malayalam, and a weak recogniser
here does active harm: it rejects good data and keeps bad.

Wire one in through the `Transcriber` protocol:

```python
from mlvoice.data import CorpusPreparer, PrepareConfig
from mlvoice.protocols import Transcriber

class IndicConformer(Transcriber):
    def transcribe(self, audio): ...

preparer = CorpusPreparer(
    PrepareConfig(output_dir=Path("data/processed"), max_asr_cer=0.15),
    transcriber=IndicConformer(),
)
```

Without a transcriber the pipeline still runs, and records the corpus as
unverified. Treat that as an unverified corpus, not a clean one.

### Splits are speaker-disjoint

A validation set sharing speakers with train measures memorisation. For a
*voice cloning* model that is precisely the quantity being faked, so the number
would look excellent and mean nothing. Split assignment hashes the speaker id,
which also makes it stable as the corpus grows.
