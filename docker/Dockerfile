# Navigator (ino) - Docker Image
# Target: Raspberry Pi 5 (linux/arm64) or x86-64 dev machine

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    ffmpeg \
    alsa-utils \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU-only first (Pi 5 / arm64 compatible)
RUN pip install --no-cache-dir torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
RUN python -c "import openwakeword.utils; openwakeword.utils.download_models()"

# Copy application code and seed data
COPY app/       ./app/
COPY prompts/   ./prompts/
COPY scripts/   ./scripts/
COPY models/    ./models/
COPY data/csv_english/ ./data/csv_english/
COPY data/csv_arabic/  ./data/csv_arabic/
COPY data/ecu_knowledge.json    ./data/ecu_knowledge.json
COPY data/ecu_knowledge_ar.json ./data/ecu_knowledge_ar.json
COPY data/corrections_en.json   ./data/corrections_en.json

# Ensure runtime directories exist
RUN mkdir -p data/sqlite logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python scripts/health_check.py || exit 1

CMD ["python", "-m", "app.main"]
