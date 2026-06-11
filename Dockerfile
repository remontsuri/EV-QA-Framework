# ============================================================
# EV-QA-Framework v2.0.0 — Production Dockerfile
# ============================================================

FROM python:3.12-slim

# Install uv (pinned for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app

# Copy everything (source + manifest)
COPY . .

# Create venv and install package in editable mode
RUN uv venv && \
    uv pip install -e . && \
    uv cache clean

# Environment
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import ev_qa_framework; assert ev_qa_framework.__version__ == '2.0.0'" || exit 1

# Default: show CLI help
ENTRYPOINT ["ev-qa"]
CMD ["--help"]
