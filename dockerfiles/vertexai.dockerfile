# ---- Base image (uv + Python 3.12) ----
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# ---- Env ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WANDB_MODE=online

# ---- System deps + gsutil ----
RUN apt update && apt install --no-install-recommends -y \
        build-essential \
        gcc \
        git \
        curl \
        gnupg \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] \
        http://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt update \
    && apt install -y google-cloud-cli \
    && apt clean && rm -rf /var/lib/apt/lists/*

# ---- Workdir ----
WORKDIR /app

# ---- Copy dependency files FIRST (Docker cache) ----
COPY pyproject.toml uv.lock ./
COPY README.md README.md
COPY src/ src/

# ---- Install Python deps ----
RUN uv sync --frozen --no-cache

# ---- Download compressed dataset at BUILD time ----
RUN gsutil cp gs://mlops-group21/compressed/data.tar.gz . \
 && tar -xzf data.tar.gz \
 && rm data.tar.gz