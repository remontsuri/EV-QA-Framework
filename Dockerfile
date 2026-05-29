# =============================================================================
# EV-QA-Framework — Production Dockerfile (Multi-stage build)
# =============================================================================
# Stage 1: Build dependencies
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install system build deps (only needed for compilation)
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        gcc \
        libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency manifests first for layer caching
COPY setup.py pyproject.toml requirements.txt README.md ./
COPY ev_qa_framework/ ev_qa_framework/

# Build wheel — install core deps (without ml extras by default)
# To enable TensorFlow/ML support, build with: --build-arg INSTALL_ML=true
ARG INSTALL_ML=false
RUN pip install --user --no-warn-script-location \
    build wheel && \
    python -m build --wheel --outdir /build/dist . && \
    if [ "$INSTALL_ML" = "true" ]; then \
        pip install --user --no-warn-script-location \
            /build/dist/*.whl[ml]; \
    else \
        pip install --user --no-warn-script-location \
            /build/dist/*.whl; \
    fi

# -----------------------------------------------------------------------------
# Stage 2: Production image (slim)
# -----------------------------------------------------------------------------
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DASHBOARD_HOST=0.0.0.0 \
    DASHBOARD_PORT=8000 \
    ENVIRONMENT=production

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY --from=builder /build/dist /tmp/dist
COPY dashboard/ dashboard/
COPY api/ api/
COPY ev_qa_framework/ ev_qa_framework/

# Ensure user-installed packages are discoverable
ENV PYTHONPATH=/root/.local/lib/python3.10/site-packages:$PYTHONPATH

# Install gunicorn for production serving
RUN pip install --no-cache-dir gunicorn==23.0.0

# Create non-root user
RUN addgroup --system --gid 1001 app && \
    adduser --system --uid 1001 --gid 1001 app && \
    chown -R app:app /app /root/.local
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8000/ || exit 1

# Run with gunicorn + uvicorn workers for production
CMD ["gunicorn", "dashboard.app:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
