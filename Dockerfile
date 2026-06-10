# ============================================================
# EV-QA-Framework v2.0.0 — Production Dockerfile
# Multi-stage build, non-root user, healthcheck
# ============================================================

# ---- Stage 1: Builder ----
FROM python:3.12-slim AS builder

# Install uv (pinned for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /build

# Copy only manifest files first → maximizes Docker layer caching
COPY pyproject.toml uv.lock ./

# Install production dependencies into a virtual environment
RUN uv sync \
    --no-dev \
    --no-extra ml \
    --frozen \
    --python python3

# ---- Stage 2: Runtime ----
FROM python:3.12-slim AS runtime

# Security: create non-root user
RUN groupadd --gid 1000 evqa && \
    useradd  --uid 1000 --gid evqa --create-home --shell /bin/bash evqa

# Install uv (lightweight, for runtime dependency management if needed)
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app

# Copy installed virtual environment from builder stage
COPY --from=builder /build/.venv /app/.venv

# Ensure the venv is on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Prevent Python from writing .pyc files (smaller layers, faster startup)
    PYTHONPATH=/app

# Copy application source code
COPY --chown=evqa:evqa . .

# Switch to non-root user
USER evqa

# Healthcheck — verifies the package is importable and version is correct
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import ev_qa_framework; assert ev_qa_framework.__version__ == '2.0.0'" || exit 1

# Default: run the framework CLI
ENTRYPOINT ["ev-qa"]
CMD ["--help"]
