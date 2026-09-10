# syntax=docker/dockerfile:1

# Multi-stage so the runtime image carries no build toolchain.
#
# The default image installs the runtime dependencies only -- the text
# frontend, the API and the CLI -- which keeps it small enough to deploy and
# start quickly. Model runtimes are multi-gigabyte and are added by the
# `models` build target, so a deployment that only needs text processing or the
# evaluation harness does not pay for CUDA wheels.

FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Runtime dependencies only. No torch: the energy VAD and every quality gate
# are pure NumPy, and resampling and loudness are core dependencies.
RUN pip install --upgrade pip && pip install "."


FROM python:3.11-slim-bookworm AS runtime

# libsndfile is needed by soundfile at runtime; nothing else is.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 mlvoice

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLVOICE_ENV=production \
    MLVOICE_LOG_JSON=true

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER mlvoice

# Writable state: the SQLite database and enrolled reference audio. Mount a
# volume here in any real deployment -- reference clips are biometric data and
# must not live in a container layer.
VOLUME ["/data"]
ENV MLVOICE_DATABASE_URL=sqlite:////data/mlvoice.db \
    MLVOICE_VOICE_STORAGE_DIR=/data/voices

EXPOSE 8000

# Readiness, not liveness: the container reports unhealthy until the backend
# has loaded, so an orchestrator does not route traffic into a cold replica.
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=4 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=4).status==200 else 1)"

CMD ["uvicorn", "mlvoice.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]


# Adds the model runtime (torch, transformers) and the evaluation extras.
# Several gigabytes larger, so it is a separate target rather than the default.
# Build with:
#   docker build -f docker/Dockerfile --target models -t mlvoice:models .
FROM runtime AS models
USER root
COPY pyproject.toml README.md LICENSE /src/
COPY src /src/src
RUN pip install --no-cache-dir "/src[models,eval]" && rm -rf /src
USER mlvoice
