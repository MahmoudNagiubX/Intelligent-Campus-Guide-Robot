import sounddevice as sd
import numpy as np

print("Listening for 5 seconds - speak now...")
recording = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='int16')
sd.wait()
peak = np.abs(recording).max()
print(f"Peak amplitude: {peak}")
if peak < 100:
    print("WARNING: mic is silent - wrong device or muted")
elif peak < 1000:
    print("WARNING: mic signal very low - check volume")
else:
    print("OK: mic is picking up audio")
