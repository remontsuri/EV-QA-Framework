FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy project manifest and lock file first (Docker layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies (skip optional ml and dev groups)
RUN uv sync --no-extra ml --no-extra dev

# Copy source code
COPY . .

# Default command: run tests with coverage
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=ev_qa_framework", "--tb=short"]
