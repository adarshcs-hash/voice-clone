# Roadmap

What this repository is, what it is not, and what it costs to finish.

## What exists

The platform a Malayalam voice model plugs into:

- A complete, tested Malayalam text frontend with no model dependency.
- A corpus preparation pipeline with quality gates, ASR verification and
  speaker-disjoint splits.
- Consent capture and verification, watermarking, moderation.
- An evaluation harness and the Malayalam hard test set.
- An HTTP service and CLI, containerised, with CI.

## What does not exist

**No model weights are trained here.** The default backend targets
`ai4bharat/IndicF5`; the `dummy` backend exists so the API contract and test
suite run on a CPU-only box.

## Phase 0 — measure (week 1, ~₹0)

Build the benchmark before building anything else. See
[`evaluation.md`](evaluation.md). Deliverable: a category-level scorecard for
every candidate system, scored by native speakers.

Nothing after this point is worth doing until this is done. It is entirely
possible that the answer is "IndicF5 with this frontend is already good enough
for our product", and that is a result worth a week.

## Phase 1 — ship on the base model (weeks 1–2, ~₹0)

```bash
MLVOICE_TTS_BACKEND=indicf5 MLVOICE_MODEL_REVISION=<sha> mlvoice serve
```

Wire in an Indic ASR model for the consent check and for evaluation, populate
the loanword lexicon and the name blocklist for your domain, and put it in front
of real users. Their complaints are better prioritisation than any plan.

## Phase 2 — fine-tune on Malayalam (weeks 3–10, ₹2–8 lakh)

The highest-value technical work, and well within a small team's reach.

1. **Assemble the corpus.** IndicVoices-R (~83 h) + IMaSC (~50 h) + IndicTTS
   (~20 h) ≈ 150 h, prepared through `mlvoice data prepare` with ASR
   verification. See [`data-and-licensing.md`](data-and-licensing.md).
2. **Record your own 20–40 h** across dialects, registers and speakers, with
   consent captured in the same session. This is the part nobody else has.
3. **Fine-tune IndicF5** on Malayalam only. LoRA or full fine-tune; 1–4 ×
   A100/4090 for 1–5 days. GPU cost is **$50–500** — the corpus is the expense,
   not the compute.
4. **Per-voice adaptation.** 30 minutes to 2 hours of one speaker as a LoRA
   gives a "professional clone" noticeably better than the zero-shot prompt.
5. **Gate on the test set,** not on training loss. Re-score with native speakers
   at each checkpoint you consider shipping.

Register the fine-tune as a backend and the rest of the system is unchanged:

```python
from mlvoice.tts.registry import register_backend
register_backend("mlvoice-ml-v1", lambda s: MyFineTune(...))
```

Expected outcome: clearly better than any general multilingual model on
Malayalam, because it is trained on 10× more Malayalam and on data nobody else
has.

## Phase 3 — production hardening (months 3–6, ₹15–40 lakh, 2–4 people)

- Postgres instead of SQLite; migrations instead of `create_all`.
- Shared-state rate limiting (Redis) — the built-in limiter is per process.
- AudioSeal watermarking.
- Streaming with real time-to-first-byte targets: a distilled or
  chunk-causal model, batched inference, KV-cache reuse.
- Object storage for reference audio, encrypted, with a retention policy.
- Metrics and tracing; alerting on real-time factor and error rate.
- A trained Manglish transliterator to replace the rule system.
- An English-to-Malayalam pronunciation model for code-mix.

## Phase 4 — only if the data justifies it

Training a foundation model from scratch needs 100k+ hours, hundreds of GPUs,
and roughly $0.5–2M. **With 150–250 hours of open Malayalam data it is not a
sensible option**, and attempting it is the most common way this kind of project
fails. Revisit only if you have acquired a corpus an order of magnitude larger
than what is described here.

## Honest cost summary

| Phase | Time | Cost | Deliverable |
|---|---|---|---|
| 0 — measure | 1 week | ~₹0 | Scorecard; a decision made on evidence |
| 1 — ship on base | 1–2 weeks | ~₹0 | A Malayalam TTS product in users' hands |
| 2 — fine-tune | 6–8 weeks | ₹2–8 lakh | A model that beats the general ones |
| 3 — harden | 3–6 months | ₹15–40 lakh | Something you can sell an SLA on |
| 4 — from scratch | 12+ months | ₹4 crore+ | Almost certainly not worth it |

## Known gaps in the current code

Tracked here rather than in comments so they are visible:

1. **Manglish transliteration is rule-based** and therefore approximate.
2. **English inside Malayalam is passed through**, not transliterated. Needs a
   pronunciation model; the lexicon is the interim answer.
3. **The rate limiter is per process** — a safety valve, not a quota.
4. **`create_all` instead of migrations.** Fine for SQLite, not for Postgres.
5. **The spread-spectrum watermark does not survive lossy compression.** Use
   AudioSeal in production.
6. **The test set is 75 cases**, not the 300+ a production gate wants.
7. **Streaming is chunk-level, not frame-level**, so time-to-first-byte is
   bounded by the first chunk's generation time.
8. **No speaker-similarity threshold calibration** is shipped; the consent
   default must be calibrated against your own verifier before it is trusted.
