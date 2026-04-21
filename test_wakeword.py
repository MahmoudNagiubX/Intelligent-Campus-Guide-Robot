import time
import numpy as np
import sounddevice as sd
from app.wakeword.detector import WakeWordDetector

detected = False

def on_activated():
    global detected
    detected = True
    print("WAKE WORD DETECTED!")

ww = WakeWordDetector(mock=False)
ww._on_activated = on_activated
ww.start() if hasattr(ww, 'start') else None

print("Say 'Hey Jarvis' now... (15 seconds)")

def callback(indata, frames, time_info, status):
    frame = indata[:, 0].astype(np.int16).tobytes()
    ww.process_frame(frame)

with sd.InputStream(samplerate=16000, channels=1, dtype='int16',
                    device=1, blocksize=512, callback=callback):
    for i in range(15):
        time.sleep(1)
        print(f"  {15-i-1}s remaining..." if not detected else "  Detected! Waiting...")
        if detected:
            break

print("Done." if detected else "Not detected - check wake word model or speak louder")
