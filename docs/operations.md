# Operating mlvoice

## Configuration

All settings are environment variables with the `MLVOICE_` prefix, validated
once at startup. `.env.example` is the full list; `config.py` is the authority.

The settings that matter most:

| Variable | Purpose |
|---|---|
| `MLVOICE_ENV` | `production` enforces the safety invariants below |
| `MLVOICE_API_KEYS` | Comma-separated accepted keys |
| `MLVOICE_TTS_BACKEND` | `indicf5` or a registered fine-tune; `dummy` for tests |
| `MLVOICE_MODEL_REVISION` | Pins the weights. Required in production |
| `MLVOICE_DEVICE` | `cpu`, `cuda` or `mps` |
| `MLVOICE_CONSENT_SIGNING_KEY` | HMAC key for consent challenge tokens |
| `MLVOICE_WATERMARK_KEY` | Watermark key. **Long-lived** — rotating it makes past audio undetectable |
| `MLVOICE_DATABASE_URL` | SQLite for development, Postgres in production |
| `MLVOICE_VOICE_STORAGE_DIR` | Where reference audio lives. Encrypt this volume |
| `MLVOICE_RATE_LIMIT_PER_MINUTE` | Per-process request budget per key |

### Production refuses to start unless

- `MLVOICE_API_KEYS` is set,
- `MLVOICE_REQUIRE_CONSENT` and `MLVOICE_WATERMARK_ENABLED` are on,
- `MLVOICE_CONSENT_SIGNING_KEY` and `MLVOICE_WATERMARK_KEY` are set,
- `MLVOICE_MODEL_REVISION` pins the weights,
- the backend is not `dummy`.

Failing at startup is deliberate. These are the invariants that make the system
safe to run, and a deployment that quietly disabled one would look identical
from the outside. The error names every problem at once, not just the first.

## Running it

```bash
mlvoice serve --host 0.0.0.0 --port 8000 --workers 4     # or uvicorn directly
docker build -f docker/Dockerfile -t mlvoice:local .     # runtime image
docker build -f docker/Dockerfile --target models -t mlvoice:models .
docker compose -f docker/docker-compose.yml up           # local stack
```

The default image has no model runtime, which keeps it small and fast to start.
The `models` target adds torch and transformers.

Running the IndicF5 backend additionally needs the `indicf5` extra, because the
model's bundled remote code imports `f5-tts` and `pydub`. `f5-tts` pulls roughly
thirty transitive dependencies -- gradio, wandb, datasets and matplotlib among
them -- which exist to serve its own demo and training tooling. Expect to trim
that for a production image: none of it is reached by the inference path, and
shipping a demo UI and a telemetry client inside a synthesis service is worth
avoiding.

## The web client

`GET /ui` serves the browser client from the API process itself: an HTML file
and a script, no build step and no CDN, so it works on an air-gapped host. It
is unauthenticated because it contains nothing but the form that collects the
API key -- every request it makes still carries `X-API-Key` and is rate-limited
like any other caller.

The page adapts to the deployment rather than showing every control it has. It
reads `auth_required`, `consent_required` and `transcription` from `GET
/v1/info` and hides what does not apply, so those three settings change what a
user sees, not just what the API accepts. `/v1/info` is therefore part of the
client contract and is served unauthenticated: a client has to be able to ask
what this deployment needs before it can ask for anything correctly.

Set `MLVOICE_UI_ENABLED=false` to switch it off. Both paths then return 404
rather than 403, so a disabled console is indistinguishable from one that was
never deployed. Consider that for an internet-facing deployment: the page is
harmless, but it does advertise that this host clones voices.

The page is served `no-store` and the script `max-age=3600`. That asymmetry is
deliberate -- a cached page outliving a redeploy would bind element ids the new
script no longer looks up, which fails as dead buttons rather than as an error.

## Transcription and startup

`MLVOICE_ASR_ENABLED` turns reference-clip transcription on. The recogniser
loads **on first use**, not at startup, and that default was bought with an
outage: on a cold cache the default Whisper model is a 3 GB download, and
pulling it inside the lifespan handler left the process stuck before it bound a
socket — no API, no web client, no `/healthz` — while an optional feature
fetched weights behind a progress bar that logged nothing.

So the first transcription is slow, and the web client says so while it waits.
Set `MLVOICE_ASR_EAGER_LOAD=true` where that trade inverts: behind a readiness
probe, a slow rollout is cheaper than a slow first request. Production with
consent enabled loads eagerly regardless and refuses to start if it cannot,
because a replica that mandates consent while unable to verify it is worse than
one that does not come up.

Either way the load logs `loading transcriber` with the model id *before* it
starts, so a stalled download is distinguishable from a hung process. To take
the download out of the request path entirely, fetch the weights ahead of time
into the same cache (`HF_HOME`) — a container image should bake them in.

## Transcription and startup

`MLVOICE_ASR_ENABLED` turns reference-clip transcription on. The recogniser
loads **on first use**, not at startup, and that default was bought with an
outage: on a cold cache the default Whisper model is a 3 GB download, and
pulling it inside the lifespan handler left the process stuck before it bound a
socket -- no API, no web client, no `/healthz` -- while an optional feature
fetched weights behind a progress bar that logged nothing.

So the first transcription is slow, and the web client says so while it waits.
Set `MLVOICE_ASR_EAGER_LOAD=true` where that trade inverts: behind a readiness
probe, a slow rollout is cheaper than a slow first request. Production with
consent enabled loads eagerly regardless and refuses to start if it cannot,
because a replica that mandates consent while unable to verify it is worse than
one that does not come up.

Either way the load logs `loading transcriber` with the model id *before* it
starts, so a stalled download is distinguishable from a hung process. To keep
the download out of the request path entirely, fetch the weights ahead of time
into the same cache (`HF_HOME`); a container image should bake them in.

## Health and rollout

| Endpoint | Meaning |
|---|---|
| `GET /healthz` | The process is alive. Never depends on the model — otherwise a slow first load gets the container killed just before it becomes useful |
| `GET /readyz` | 503 until the backend has loaded. Use this as the readiness probe so a rolling deploy does not route into a cold replica |
| `GET /v1/info` | The running configuration: backend, model id, revision, whether consent and watermarking are on |

Startup also runs one synthesis through the whole chain (`warm-up complete` in
the log). Several dependencies import lazily -- `pyloudnorm` pulls in SciPy, and
NumPy plans its first FFT -- and without this the first *user* request paid for
it: measured at 1.0 s on Linux, and appreciably worse on macOS where the first
load of a signed dylib also goes through Gatekeeper. A warm-up failure is
logged but never blocks startup.

The model loads during application startup, so a weight-loading failure stops
the rollout instead of surfacing as a 500 on the first real request. With
`trust_remote_code` models, expect the first start on a cold cache to take
minutes; the container health check allows a 90-second start period, and you
should bake weights into the image or warm a shared cache volume in production.

## What to watch

Every log line is one JSON object and carries the `request_id`.

| Signal | Source | Why |
|---|---|---|
| `real_time_factor` | `X-Real-Time-Factor`, and the `synthesis served` log line | The cost and capacity signal. Above ~0.5 on a GPU, something is wrong |
| 429 rate | Access log | Distinguishes a runaway client from real growth |
| `moderation` severity | `synthesis served` log line | `flag` volume is your abuse trend |
| `watermarking skipped` | Warning log | Clips too short to mark. Routine warnings mean you have lost the provenance trail |
| `legacy_nta_fixed` | `synthesis served`, text summary | High counts mean an upstream system is emitting legacy encodings — worth fixing at source |
| `/readyz` failures | Probe | Model load problems |

### Tracing one complaint

A user reports a mispronunciation. Take the `X-Request-Id` from their response
(or find the request in the access log), then:

```bash
curl -sX POST localhost:8000/v1/text/analyze \
  -H 'X-API-Key: ...' -H 'content-type: application/json' \
  -d '{"text":"<the text they sent>"}' | jq
```

The response shows the Unicode normalisation applied, the expanded numbers and
dates, the code-mix routing, the chunk boundaries and the phoneme string per
chunk. Most pronunciation complaints are resolved here, in the frontend, without
touching the model.

## Scaling

- **Stateless replicas.** All shared state is the database and the reference
  audio directory. Point both at shared infrastructure and scale horizontally.
- **Rate limiting is per process.** With `n` replicas the effective limit is
  `n × rate`. Move to Redis or your gateway's limiter for a real quota;
  `RateLimiter` is a protocol so the swap is a constructor change.
- **The database.** Use Postgres and run migrations; pass
  `create_tables=False` to `VoiceStore` so a replica cannot race schema
  creation.
- **GPU utilisation.** One request occupies the model for seconds. Batch across
  requests, or run several single-request workers behind a queue — do not raise
  `--workers` on one GPU and hope.
- **Reference audio.** Object storage rather than a shared filesystem, with
  encryption at rest and a retention policy. It is biometric data.

## Runbook

**Model will not load.** Check `/v1/info` for the revision and the logs for
`backend_unavailable`; the error carries a `hint` for the common causes. In
order of likelihood: the repository is gated and the process has no token for
an account with access (401 / `GatedRepoError` — set `HF_TOKEN`); `transformers`
is 4.50 or newer, which makes meta-device initialisation unconditional so the
model's bundled vocoder cannot be moved to a device ("Cannot copy out of meta
tensor" — install `transformers>=4.44,<4.50`); `f5_tts` is the PyPI package
rather than AI4Bharat's fork ("load_model() missing 1 required positional
argument" — install the `indicf5` extra, which replaces it); the `indicf5`
extra is missing entirely, so `f5_tts` and `pydub` are absent; a bad revision
pin; or no disk space for the weights cache.

**Latency has regressed.** Compare `X-Real-Time-Factor` against your baseline.
If chunk count per request has grown, someone is sending longer text; the
per-request character budget is the lever.

**A voice must be taken down now.** `POST /v1/consents/{id}/revoke` disables
every voice that relied on that consent immediately. `DELETE
/v1/voices/{id}` also removes the audio from disk.

**"Did you generate this clip?"** `POST /v1/watermark/detect` with the audio.
A positive detection returns the payload, which is a key into the request log.
A negative result on re-encoded audio is inconclusive with the spread-spectrum
watermark — it does not survive lossy compression.

**Disk is full.** Reference audio under `MLVOICE_VOICE_STORAGE_DIR` grows with
enrolments. It has no automatic expiry by design: silently deleting someone's
enrolled voice is worse than running out of disk. Apply a retention policy
deliberately.
