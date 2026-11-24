# syntax=docker/dockerfile:1.9
#
# The pipeline needs three Python environments that cannot share an interpreter:
# the project runs on 3.14, while whisperx and pyannote.audio each pin a
# different torch and refuse 3.14. On a laptop those are `.venv-align` and
# `.venv-diarize`, built against whatever ffmpeg happens to be installed. That
# coupling is what breaks: pyannote's decoder, torchcodec, ships prebuilt
# libraries linked to one ffmpeg ABI at a hard-coded rpath, so a host ffmpeg
# upgrade silently disables local diarization.
#
# This image pins ffmpeg, both Pythons, and both torch stacks together, and the
# workers decode through the ffmpeg CLI rather than torchcodec, so there is no
# ABI to drift. The two worker environments build as independent stages so their
# torch downloads run in parallel and neither can perturb the other's pin.

FROM ghcr.io/astral-sh/uv:0.11.2 AS uv


FROM python:3.14-slim-trixie AS base

COPY --from=uv /uv /usr/local/bin/uv

# ffmpeg is the pipeline's only native dependency, and now its only audio
# decoder. Pinning it here is the point of the image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

# Installed in the shared base so the venvs copied from the worker stages find
# the interpreter they were built against.
RUN uv python install 3.12


# --- worker environments, built in parallel ------------------------------------

FROM base AS align-env
# CPU wheels only: this image targets Docker Desktop and CPU hosts, and the CUDA
# builds are several gigabytes each.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.12 /opt/venvs/align \
 && uv pip install --python /opt/venvs/align/bin/python --torch-backend=cpu whisperx


FROM base AS diarize-env
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.12 /opt/venvs/diarize \
 && uv pip install --python /opt/venvs/diarize/bin/python --torch-backend=cpu pyannote.audio


# --- runtime -------------------------------------------------------------------

FROM base AS runtime

COPY --from=align-env /opt/venvs/align /opt/venvs/align
COPY --from=diarize-env /opt/venvs/diarize /opt/venvs/diarize

WORKDIR /app

# Dependencies resolve from the lockfile before the source is copied, so editing
# a module does not reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

COPY transcribe_pipeline/ transcribe_pipeline/
COPY workers/ workers/
COPY scripts/ scripts/
COPY skills/ skills/
COPY bin/ bin/
COPY tests/ tests/
COPY conftest.py transcribe.py transcribe_cli.py main.py ./

# The worker environments live outside /app so a bind-mounted checkout cannot
# shadow them. These overrides are what point the pipeline, `check-setup`, and
# both setup scripts at them.
ENV TRANSCRIBE_ALIGN_VENV=/opt/venvs/align \
    TRANSCRIBE_DIARIZE_VENV=/opt/venvs/diarize \
    PATH="/app/.venv/bin:${PATH}" \
    HOME=/home/transcribe \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    MPLCONFIGDIR=/cache/matplotlib \
    TRANSCRIBE_JOBS_DIR=/cache/jobs

# Model weights are not baked in: Community-1 is gated, so shipping it would
# redistribute a licensed model and would need a build-time token. They land in
# /cache instead, which compose keeps in a named volume across runs. Everything
# the run writes to — cache, job state, transcripts — is owned by the runtime
# user; the code and the environments stay read-only to it.
RUN useradd --create-home --home-dir /home/transcribe --uid 1000 transcribe \
 && install -d -o transcribe -g transcribe /cache /cache/jobs /app/output /app/input

USER transcribe

ENTRYPOINT ["/app/bin/transcribe"]
CMD ["check-setup"]
