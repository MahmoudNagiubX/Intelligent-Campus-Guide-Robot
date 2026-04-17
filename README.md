# Navigator — Streaming Campus Guide Robot

An intelligent, voice-first campus guide robot.

**Listen → Understand → Retrieve → Verify → Answer**

## Stack
| Layer | Choice |
|---|---|
| Wake word | openWakeWord |
| VAD | Silero VAD |
| STT | Deepgram Nova-3 |
| Orchestration | Pipecat |
| LLM / Router | Groq llama-3.1-8b-instant |
| Truth system | SQLite + FTS5 + aliases |
| TTS | edge-tts |
| Hardware | Raspberry Pi 5 (8 GB) |

## Quick start

```bash
cp .env.example .env
# Fill in .env values
make install
make run
```

## Run tests
```bash
make test
```
