# ---- Base image (uv + Python 3.12) ----
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# ---- Env (dobro za ML + logs) ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WANDB_MODE=online

# ---- System deps ----
RUN apt update && \
    apt install --no-install-recommends -y \
        build-essential \
        gcc \
        git \
        curl \
    && apt clean && rm -rf /var/lib/apt/lists/*

# ---- Workdir ----
WORKDIR /app

# ---- Copy dependency files FIRST (Docker cache) ----
COPY pyproject.toml uv.lock ./

# ---- Install Python deps ----
RUN uv sync --frozen --no-cache

# ---- Copy rest of the project ----
COPY . .

# ---- Pull data with DVC (uses Cloud Build service account ADC) ----
RUN uv run dvc pull

# ---- Run training ----
ENTRYPOINT ["uv", "run", "src/eurosat/train.py"]

