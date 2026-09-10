# Safety and compliance

Voice cloning has a small number of high-volume abuse patterns, and Malayalam
has a local version of each: political deepfakes around Kerala elections, cloned
film-actor voices, and above all the "relative in trouble, send money now" phone
scam — devastatingly effective in a language where the listener has never heard
synthetic Malayalam before.

The controls below are enforced in code. `MLVOICE_ENV=production` refuses to
start if any of them is disabled, because a deployment that quietly turned one
off would look identical from the outside.

## 1. Consent

### The mechanism

Enrolment is a two-step spoken challenge (`voices/consent.py`,
`voices/enrollment.py`):

1. The service issues a phrase containing the subject's name, today's date and
   a random nonce drawn from common Malayalam words, plus an HMAC-signed token.
2. The subject records themselves reading it.
3. The service verifies **both** that the phrase was spoken (ASR, gated on CER
   and on the nonce words appearing) **and** that it is the same speaker as the
   reference clip (speaker verification, gated on cosine similarity).

### Why each part is there

- **Two steps, not one.** A single-call enrolment cannot establish consent: any
  audio could be supplied for both roles.
- **The nonce.** Without it, a recording of someone saying "I consent" could be
  lifted from anywhere. With it, the recording must have been made after the
  challenge was issued, for this enrolment.
- **The signed token.** Challenges are not server-side state, so enrolment works
  across replicas, and a client cannot forge or replay one. Tokens expire after
  an hour.
- **The same-speaker check.** Without it, anyone can read the challenge on the
  subject's behalf.
- **Re-checked at synthesis time.** A consent record can be revoked after a
  voice is created; every request re-reads the voice's status and its consent
  record.

### What it is not

It proves the same person recorded both clips. It does **not** prove who that
person is. Binding to a legal identity needs a document or government-ID check,
which belongs at a higher layer.

### Rights that must work

| Right | Implementation |
|---|---|
| Revoke consent | `POST /v1/consents/{id}/revoke` — cascades to disable every voice that relied on it |
| Delete the voice | `DELETE /v1/voices/{id}` — removes the row, the consent record, the reference audio and the consent audio from disk |
| Expiry | Consent records carry `expires_at`, default one year; an expired record fails the synthesis-time check |

Deletion means the recording is gone. A voice is biometric data; a "delete" that
leaves the audio on disk is not a deletion.

### India's DPDP Act

The Digital Personal Data Protection Act, 2023 governs this directly: a voice
recording is personal data, and a voiceprint used to identify or imitate someone
is sensitive in practice whatever the taxonomy says. What this project gives
you, and what it does not:

**Provided:** an itemised, auditable consent record per voice (who, to what,
when, how verified); revocation that takes effect; deletion that removes the
data; and no logging of raw audio or of API keys.

**Your responsibility:** a consent notice in the language the subject
understands — for a Malayalam product, in Malayalam; a named grievance-redressal
contact; breach notification; data-retention limits; and, if you process
children's data, the additional restrictions. Take legal advice. This document
is engineering notes, not a compliance opinion.

## 2. Provenance: watermarking

Every clip from `POST /v1/tts` carries a watermark whose payload is a key into
the request log. `POST /v1/watermark/detect` answers "did we generate this?",
which is what makes an abuse report actionable.

| Implementation | Robustness | Use |
|---|---|---|
| **AudioSeal** | Survives MP3/AAC, resampling, filtering, cropping | What production should run |
| **Spread-spectrum** (built in) | Survives gain changes and lossless format conversion. **Does not survive lossy compression, aggressive resampling or time-stretching** | Default, so a stock install still watermarks |

The default is honest about its limits rather than hiding them. If your audio
will be re-encoded before it reaches an adversary — and on any social platform
it will be — use AudioSeal.

Two operational rules:

- **The watermark key is long-lived.** Rotating it makes previously generated
  audio undetectable. Store it like a signing key, not like a config value.
- **Streamed audio is not watermarked.** The payload needs the whole utterance.
  `POST /v1/tts/stream` says so in `X-Watermarked: false`. If you need a
  provenance-marked artefact, use the non-streaming endpoint.

## 3. Moderation

`safety/moderation.py` is the cheap deterministic first layer:

- **Fraud templates** in Malayalam and English: OTP and credential
  solicitation, urgent money transfer, the relative-in-trouble script, lottery
  scams, authority impersonation. Matching is proximity-based and
  order-independent, because a fraud script says both "share your OTP" and
  "OTP അയക്കൂ".
- **Severity, not a binary.** Only `BLOCK` refuses; `FLAG` allows and marks for
  review. Refusing a legitimate audiobook because it contains the word "OTP" is
  its own failure.
- **A name blocklist** at enrolment, for public figures. Substring-based on a
  folded form, so "CM Pinarayi Vijayan (test)" matches. Populate it with Kerala
  politicians and film actors via `MLVOICE_BLOCKED_VOICE_NAMES_FILE`.

None of this stops a determined actor. A pattern list catches the lazy attempt
and raises the cost of the rest; the layers that actually matter are
identity-verified enrolment, watermarking and rate limiting.
`PatternModerator` implements a protocol so a trained classifier can replace it
without touching the call sites.

## 4. Access control and abuse limits

- **API keys** compared in constant time; the owner id is derived from the key
  by hash, so one tenant's voices are invisible to another. A request for
  another tenant's voice returns 404, not 403 — it must not confirm that the
  voice exists.
- **Rate limiting** over both requests and characters, so one caller sending
  novels cannot starve everyone else while staying under a request-count limit.
  Per process: with several replicas the effective limit is `replicas × rate`.
  It is a safety valve, not a quota; a quota needs shared state.
- **Reference audio uploads** are capped and quality-gated.
- **Nothing sensitive is logged.** No raw audio, no API keys, no reference
  transcripts. Unexpected server errors return a generic message rather than the
  exception text, which can carry file paths and reference-audio details.

## 5. Deployment checklist

- [ ] `MLVOICE_ENV=production` (this alone enforces the invariants below)
- [ ] `MLVOICE_API_KEYS` set; keys distributed out of band
- [ ] `MLVOICE_CONSENT_SIGNING_KEY` and `MLVOICE_WATERMARK_KEY` generated with a
      CSPRNG and stored in a secret manager
- [ ] `MLVOICE_MODEL_REVISION` pins the weights (`trust_remote_code` executes
      code from that revision)
- [ ] AudioSeal configured if output will be re-encoded anywhere
- [ ] `MLVOICE_BLOCKED_VOICE_NAMES_FILE` populated for your jurisdiction
- [ ] Reference audio on an encrypted volume with a retention policy
- [ ] Consent notice published in Malayalam
- [ ] A grievance contact and an abuse-report path that reaches a human
- [ ] Someone on call who can revoke a consent record and disable a voice
