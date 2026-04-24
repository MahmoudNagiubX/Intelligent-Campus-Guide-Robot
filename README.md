# ino - Streaming Campus Guide Robot

ino is a voice-first campus guide robot for ECU.

Pipeline: wake word -> VAD -> STT -> router -> RAG retrieval -> grounded answer -> TTS -> speaker playback.

## Stack

| Layer | Choice |
|---|---|
| Wake word | openWakeWord |
| VAD | Silero VAD |
| STT | Deepgram Nova-3 |
| Arabic STT | ElevenLabs Scribe v2, optional |
| Orchestration | Pipecat |
| LLM/router | Groq |
| Truth system | SQLite + FTS5 + CSV sync |
| TTS | edge-tts |
| Audio I/O | PyAudio input + sounddevice output |

## Quick Start

Use Python 3.11 or 3.12.

```powershell
git clone <repo-url>
cd "Robot Brain & Voice System"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and fill these required English-mode keys:

```env
DEEPGRAM_API_KEY=your_deepgram_key_here
GROQ_API_KEY=your_groq_key_here
ENGLISH_ONLY_MODE=true
```

Then run:

```powershell
python -m scripts.health_check
python -m app.main
```

Say: `Hey Jarvis, where is the robotics lab?`

## Arabic Mode

Arabic STT is optional. To enable it, set:

```env
ENGLISH_ONLY_MODE=false
ELEVENLABS_API_KEY=your_elevenlabs_key_here
```

If ElevenLabs returns repeated 403 errors, ino permanently skips it for the current process and falls back to Deepgram multi-language mode.

## Common Commands

```powershell
python -m app.main
python -m scripts.health_check
python -m scripts.smoke_test
python -m pytest -q
```

The Makefile also provides `make install`, `make run`, and `make test` on systems with `make`.

## Hardware Notes

Set `MIC_DEVICE_INDEX=-1` and `SPEAKER_DEVICE_INDEX=` to use system defaults. If the wrong device is selected, change these values in `.env`.

ino wakes on `Hey ino` using the trained openWakeWord model at `models/hey_ino.onnx`. Keep `WAKE_WORD_MODEL=models/hey_ino.onnx` in `.env`, or change it only if you replace the wake-word model file.

PyAudio may need system audio headers. On Windows, install a matching PyAudio wheel or Microsoft C++ Build Tools if `pip install -r requirements.txt` fails. On Raspberry Pi/Linux, install PortAudio first, then reinstall requirements.
