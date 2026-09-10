# Security policy

## Reporting a vulnerability

Report privately, not as a public issue. Include a reproduction and the impact
you believe it has; expect an acknowledgement within a few working days.

## What is in scope

This is a voice-cloning system, so the security-relevant surface is wider than
usual. Treat any of the following as a vulnerability:

- **Bypassing consent.** Forging or replaying a consent challenge token,
  enrolling a voice without a verified consent record, or synthesising with a
  voice whose consent has been revoked or has expired.
- **Cross-tenant access.** Reading, using or deleting another API key's voices,
  or confirming that another tenant's voice exists.
- **Provenance failures.** Suppressing the watermark on `POST /v1/tts`,
  forging a detection, or recovering a payload without the deployment key.
- **Leakage.** Reference audio, consent recordings, API keys, or reference
  transcripts appearing in logs, error bodies, or API responses.
- **Refusal bypass** that gets clearly prohibited content through the moderation
  layer in bulk.
- The usual: injection, path traversal (particularly through uploaded filenames
  and voice ids), resource exhaustion, and dependency vulnerabilities.

## Known limitations, not vulnerabilities

These are documented design limits. Reports about them are welcome as issues,
but they are not security reports:

- The **spread-spectrum watermark does not survive lossy compression**. Use
  AudioSeal where that matters. See
  [`docs/safety-and-compliance.md`](docs/safety-and-compliance.md).
- The **rate limiter is per process**, so with `n` replicas the effective limit
  is `n × rate`.
- **Pattern-based moderation is a first layer.** It catches templates, not
  determined actors.
- **Consent verification is not identity verification.** It proves the same
  person recorded the reference and the consent clip, not who that person is.
- `MLVOICE_ENV=development` deliberately allows unsafe combinations (no auth, no
  consent, no watermark) so the code paths can be tested. Production refuses
  them.
- `trust_remote_code=True` is required by the IndicF5 family and is remote code
  execution by design. Pin `MLVOICE_MODEL_REVISION`.

## Handling reference audio

Enrolled reference clips and consent recordings are biometric data. If you
deploy this, encrypt the volume, set a retention policy, and make sure
`DELETE /v1/voices/{id}` is reachable by the people who need it.
