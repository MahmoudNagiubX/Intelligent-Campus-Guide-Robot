# Navigator - Raspberry Pi 5 Setup Guide
## Phase 10, Step 10.2

This document walks a hardware engineer or developer through setting up Navigator
on a fresh Raspberry Pi 5 (8 GB) running Raspberry Pi OS.

---

## 1. Hardware requirements

| Component | Specification |
|---|---|
| Board | Raspberry Pi 5 (8 GB RAM) |
| OS | Raspberry Pi OS 64-bit (bookworm, headless) |
| Storage | 32 GB+ microSD or NVMe SSD |
| Microphone | USB microphone (16 kHz capable) or USB audio adapter with lapel mic |
| Speaker | USB speaker or 3.5 mm audio output via USB audio adapter |
| Network | Ethernet or Wi-Fi (stable internet required for Deepgram + Groq) |

---

## 2. OS installation

1. Flash **Raspberry Pi OS Lite (64-bit)** using Raspberry Pi Imager.
2. Enable SSH and set hostname/Wi-Fi credentials in Imager settings before flashing.
3. Boot the Pi and connect via SSH.

---

## 3. System dependencies

```bash
sudo apt-get update && sudo apt-get upgrade -y

# Audio libraries
sudo apt-get install -y \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    ffmpeg \
    alsa-utils

# Git
sudo apt-get install -y git

# Python 3.11 (Bookworm ships 3.11)
python3 --version   # confirm: Python 3.11.x
```

---

## 4. Audio setup

### 4.1 List audio devices
```bash
aplay -l   # list playback devices
arecord -l # list capture devices
```

### 4.2 Set USB microphone as default input
Create or edit `~/.asoundrc`:
```
pcm.!default {
    type asym
    capture.pcm "mic"
    playback.pcm "speaker"
}
pcm.mic {
    type plug
    slave { pcm "hw:1,0" }   # adjust card number from arecord -l
}
pcm.speaker {
    type plug
    slave { pcm "hw:0,0" }   # adjust card number from aplay -l
}
```

### 4.3 Test microphone
```bash
arecord -d 3 -f S16_LE -r 16000 -c 1 test.wav
aplay test.wav
```
You should hear your own voice played back.

---

## 5. Python environment

```bash
# Install pip and venv
sudo apt-get install -y python3-pip python3-venv

# Clone the repository
git clone https://github.com/MahmoudNagiubX/Intelligent-Campus-Guide-Robot.git navigator
cd navigator

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note**: PyAudio requires `portaudio19-dev` to compile. If installation fails, run:
> ```bash
> sudo apt-get install -y python3-pyaudio
> ```
> and import via the system package instead.

---

## 6. Environment configuration

```bash
cp .env.example .env
nano .env
```

Fill in:
```
DEEPGRAM_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
MIC_SAMPLE_RATE=16000
MIC_DEVICE_INDEX=1        # set from arecord -l, usually 1 for USB mic
SQLITE_DB_PATH=./data/sqlite/navigator.db
CSV_DATA_DIR=./data/csv
EDGE_TTS_VOICE_AR=ar-EG-SalmaNeural
EDGE_TTS_VOICE_EN=en-US-JennyNeural
ACTION_BRIDGE_URL=http://localhost:8765/navigate
LOG_LEVEL=INFO
```

---

## 7. Startup sequence

The navigator application follows this boot sequence automatically:

1. Load and validate `config` (all env vars present)
2. Open `SQLite` database (create if not exists)
3. Run `bootstrap_schema()` — create all tables and FTS indexes
4. Run `sync_all_csvs()` — import CSV data into SQLite
5. Initialize `MicCapture` — open audio device
6. Initialize `WakeWordDetector` — load openWakeWord model
7. Initialize `SileroVAD` — load Silero model from torch hub
8. Initialize `DeepgramStreamingClient` — establish WebSocket
9. Initialize `GroqClient` — verify API key
10. Initialize `EdgeTTSClient` + `PlaybackManager`
11. Initialize `ConversationController`
12. Initialize `NavigationBridge` (action bridge)
13. Enter **IDLE** state — await wake word

```bash
# Manual start
source .venv/bin/activate
python -m app.main
```

---

## 8. Health checks

Run the health check script:
```bash
python scripts/health_check.py
```

Checks performed:
- ✅ Config loads (all required env vars present)
- ✅ Microphone device reachable
- ✅ SQLite opens and schema exists
- ✅ CSV sync completed without errors
- ✅ Internet reachable (ping api.groq.com)
- ✅ Groq and Deepgram keys valid format

---

## 9. Docker alternative

If running Docker on Pi:
```bash
# Install Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# Start Navigator
cd navigator/docker
docker compose up --build
```

Audio passthrough inside Docker on Pi requires the `/dev/snd` device mount
(already configured in `docker-compose.yml`).

---

## 10. Auto-start on boot

To run Navigator automatically after boot:
```bash
sudo nano /etc/systemd/system/navigator.service
```

```ini
[Unit]
Description=Navigator Campus Guide Robot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/navigator
ExecStart=/home/pi/navigator/.venv/bin/python -m app.main
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable navigator
sudo systemctl start navigator
sudo systemctl status navigator
```

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named pyaudio` | `sudo apt-get install python3-pyaudio` |
| Mic device not found | Check `arecord -l` and set `MIC_DEVICE_INDEX` in `.env` |
| Silero VAD slow to load | First boot downloads the model — allow 30–60 seconds |
| Deepgram `Unauthorized` | Check `DEEPGRAM_API_KEY` in `.env` |
| Wake word never fires | Test mic with `arecord` first, then check OWW model path |
| TTS no audio | Check speaker connection, `aplay -l`, and ALSA config |
