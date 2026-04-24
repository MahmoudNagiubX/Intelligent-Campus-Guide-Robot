# Navigator (ino) - Raspberry Pi 5 Deployment Guide

## Hardware requirements

| Component | Requirement |
|---|---|
| Board | Raspberry Pi 5, 8 GB RAM |
| OS | Raspberry Pi OS 64-bit Lite (Bookworm) |
| Storage | 32 GB+ microSD or NVMe SSD |
| Microphone | USB microphone, 16 kHz capable |
| Speaker | USB speaker or 3.5 mm via USB audio adapter |
| Network | Stable Wi-Fi or Ethernet (internet required) |

---

## Step 1 - Flash OS

Use Raspberry Pi Imager. Select **Raspberry Pi OS Lite (64-bit)**.
Enable SSH and set hostname/Wi-Fi in Imager before flashing.

---

## Step 2 - System Packages

```bash
sudo apt-get update && sudo apt-get upgrade -y

sudo apt-get install -y \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    ffmpeg \
    alsa-utils \
    git \
    python3-pip \
    python3-venv
```

---

## Step 3 - Audio Device Setup

```bash
# List playback and capture devices
aplay -l
arecord -l
```

Note the card number for your USB microphone and speaker.
Create `~/.asoundrc`:

```text
pcm.!default {
    type asym
    capture.pcm "mic"
    playback.pcm "speaker"
}
pcm.mic {
    type plug
    slave { pcm "hw:1,0" }
}
pcm.speaker {
    type plug
    slave { pcm "hw:0,0" }
}
```

Replace `hw:1,0` and `hw:0,0` with your actual card numbers.

Test:

```bash
arecord -d 3 -f S16_LE -r 16000 -c 1 test.wav && aplay test.wav
```

---

## Step 4 - Clone And Install

```bash
git clone <your-repo-url> navigator
cd navigator

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

# Install CPU-only PyTorch first (avoids downloading the GPU version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install everything else
pip install -r requirements.txt
```

---

## Step 5 - Configure

```bash
cp .env.example .env
nano .env
```

Fill in:

```env
DEEPGRAM_API_KEY=your_deepgram_key
ELEVENLABS_API_KEY=your_elevenlabs_key
GROQ_API_KEY=your_groq_key

MIC_DEVICE_INDEX=1
SPEAKER_DEVICE_INDEX=0

ENGLISH_ONLY_MODE=true
SESSION_TIMEOUT_SEC=25
EDGE_TTS_VOICE_EN=en-US-ChristopherNeural
EDGE_TTS_VOICE_AR=ar-EG-SalmaNeural
EDGE_TTS_RATE=-10%
EDGE_TTS_RATE_AR=-3%

WAKE_WORD=hey ino
WAKE_WORD_MODEL=models/hey_ino.onnx

LOG_LEVEL=INFO
SQLITE_DB_PATH=./data/sqlite/navigator.db
CSV_ENGLISH_DIR=./data/csv_english
CSV_ARABIC_DIR=./data/csv_arabic
```

---

## Step 6 - Run Health Check

```bash
python scripts/health_check.py
```

All checks must pass before starting the robot.

---

## Step 7 - Run The Robot

```bash
python -m app.main
```

The system will:

1. Sync CSV data to SQLite.
2. Run health checks and abort if any fail.
3. Pre-warm the Groq connection.
4. Pre-synthesize TTS fallback audio.
5. Load the wake word model.
6. Enter idle listening mode, waiting for **"Hey ino"**.

---

## Step 8 - Auto-Start On Boot (Optional)

Create `/etc/systemd/system/navigator.service`:

```ini
[Unit]
Description=Navigator Campus Guide Robot
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/navigator
Environment=PATH=/home/pi/navigator/.venv/bin:/usr/bin:/bin
ExecStart=/home/pi/navigator/.venv/bin/python -m app.main
Restart=on-failure
RestartSec=5
StandardOutput=append:/home/pi/navigator/logs/navigator.log
StandardError=append:/home/pi/navigator/logs/navigator.log

[Install]
WantedBy=multi-user.target
```

Enable:

```bash
mkdir -p /home/pi/navigator/logs
sudo systemctl daemon-reload
sudo systemctl enable navigator
sudo systemctl start navigator
sudo systemctl status navigator
```

---

## Troubleshooting

**"No audio was received" from TTS** - edge-tts network issue. The robot has a retry mechanism. If it persists, check internet connectivity.

**"DEEPGRAM_API_KEY is missing"** - `.env` is not loaded. Ensure you ran `cp .env.example .env` and filled in the keys.

**Wake word not triggering** - check `MIC_DEVICE_INDEX`, confirm `models/hey_ino.onnx` exists, run `arecord -l`, and try index 1 or 2.

**Robot responds but no audio** - check `SPEAKER_DEVICE_INDEX`. Run `aplay -l` and try different values.

**High CPU usage** - normal during VAD model load, especially on first startup. It settles after the Silero model warms up.
