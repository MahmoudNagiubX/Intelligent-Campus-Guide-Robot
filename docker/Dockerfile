# Navigator - Docker Image
# Phase 10, Step 10.1
# Target: Raspberry Pi 5 (arm64) or x86-64 dev machine

FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
# PyAudio requires portaudio; sounddevice requires libasound for ALSA playback
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies first (layer caching) ─────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──────────────────────────────────────────────────────
COPY app/       ./app/
COPY prompts/   ./prompts/
COPY scripts/   ./scripts/

# ── Ensure data directories exist (volumes mounted at runtime) ─────────────────
RUN mkdir -p data/csv data/sqlite data/synonyms logs

# ── Environment defaults (override via .env or docker-compose env_file) ───────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=INFO

# ── Health check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "from app.config import get_settings; get_settings(); print('OK')" || exit 1

# ── Startup ────────────────────────────────────────────────────────────────────
CMD ["python", "-m", "app.main"]
