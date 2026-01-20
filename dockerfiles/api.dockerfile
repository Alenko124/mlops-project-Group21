# ---- Base image (uv + Python 3.12) ----
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

EXPOSE 8080


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080


RUN apt update && apt install --no-install-recommends -y \
        build-essential \
        gcc \
        curl \
    && apt clean && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY pyproject.toml uv.lock README.md ./
COPY src/ src/


RUN uv sync --frozen --no-cache


CMD ["uv", "run", "python", "-m", "uvicorn", "src.eurosat.api:app", "--host", "0.0.0.0", "--port", "8080"]
