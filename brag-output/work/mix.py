# Narration over the score: audio.wav (music+SFX) + vo/<n>.wav → audio-vo.wav, music ducked under the voice.
import json, subprocess
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, butter, sosfilt

SR = 44100
LINES = json.loads(subprocess.check_output(["node", "-e", "console.log(JSON.stringify(require('./vo.js')))"]))

sr, music = wavfile.read("audio.wav")
music = music.astype(float) / 32767
N = len(music)
voice = np.zeros(N)
hp = butter(2, 90, "high", fs=SR, output="sos")

for i, (start, end, text) in enumerate(LINES):
    vsr, v = wavfile.read(f"vo/{i}.wav")
    v = resample_poly(v.astype(float), SR, vsr)
    env = np.convolve(np.abs(v), np.ones(441) / 441, "same")
    on = np.where(env > 0.02 * env.max())[0]
    v = v[max(0, on[0] - 800): on[-1] + 4000]          # trim silence, keep a breath of tail
    v = sosfilt(hp, v)
    v *= 0.22 / np.sqrt(np.mean(v ** 2))                # same loudness for every line
    speech = (on[-1] - on[0]) / SR
    assert start + speech <= end + 0.05, f"line {i} overruns its scene: {speech:.2f}s"
    a = int(start * SR) - 800
    voice[a:a + len(v)] += v[: N - a]

# duck: smooth voice activity → music gain down ~7 dB while she speaks
act = np.convolve((np.abs(voice) > 0.01).astype(float), np.ones(int(.12 * SR)) / int(.12 * SR), "same")
act = np.convolve(np.clip(act * 4, 0, 1), np.ones(int(.25 * SR)) / int(.25 * SR), "same")
duck = 1 - 0.55 * act

mix = music * duck[:, None] + voice[:, None] * 0.9
mix = np.tanh(mix * 1.2) / np.tanh(1.2)
mix *= 0.89 / np.abs(mix).max()
wavfile.write("audio-vo.wav", SR, (mix * 32767).astype(np.int16))
print("ok")
