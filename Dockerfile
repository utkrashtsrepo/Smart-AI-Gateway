# Multi-stage lightweight Dockerfile for Smart AI Gateway
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

# Copy dependency definition
COPY pyproject.toml .

# Install dependencies into virtualenv
RUN uv venv /opt/venv && \
    uv pip install --no-cache -r pyproject.toml

# Final runtime image
FROM python:3.12-slim AS runner

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy virtualenv and application code
COPY --from=builder /opt/venv /opt/venv
COPY . .

# Ensure data directory exists for SQLite
RUN mkdir -p /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

