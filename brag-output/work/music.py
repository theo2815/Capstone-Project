# 23s score + SFX for the QuickPitik brag. 120 BPM, A major. Writes audio.wav.
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile

SR, DUR, BEAT = 44100, 23.0, 0.5
N = int(SR * DUR)
rng = np.random.default_rng(3)
music = np.zeros((N, 2)); send = np.zeros((N, 2))

def hz(m): return 440 * 2 ** ((m - 69) / 12)
def lp(x, f, o=2): return sosfilt(butter(o, f, "low", fs=SR, output="sos"), x)
def hp(x, f, o=2): return sosfilt(butter(o, f, "high", fs=SR, output="sos"), x)
def bp(x, lo, hi): return sosfilt(butter(2, [lo, hi], "band", fs=SR, output="sos"), x)
def env(n, a=0.005, d=0.3):
    t = np.arange(n) / SR
    return np.minimum(t / a, 1) * np.exp(-t / d)
def add(buf, x, at, gain=1.0, pan=0.0, rev=0.0):
    i = int(at * SR); x = x[: max(0, N - i)]
    l, r = gain * np.sqrt(0.5 * (1 - pan)), gain * np.sqrt(0.5 * (1 + pan))
    buf[i:i + len(x), 0] += x * l; buf[i:i + len(x), 1] += x * r
    if rev: send[i:i + len(x), 0] += x * l * rev; send[i:i + len(x), 1] += x * r * rev
def saw(f, n, det=0.0):
    t = np.arange(n) / SR
    return sum(2 * ((t * f * (1 + d)) % 1) - 1 for d in (-det, 0, det)) / 3

# chords per 2s bar: A  F#m  D  E  (MIDI roots)
CH = [(57, [0, 4, 7]), (54, [0, 3, 7]), (50, [0, 4, 7]), (52, [0, 4, 7])]
def chord(t): return CH[int(t // 2) % 4]

# ── instruments ──
def kick(g):
    n = int(.35 * SR); t = np.arange(n) / SR
    f = 45 + 90 * np.exp(-t / .03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, .002, .12) * g
def hat():
    n = int(.05 * SR); return hp(rng.standard_normal(n), 7000) * env(n, .001, .012)
def clap():
    n = int(.25 * SR); x = bp(rng.standard_normal(n), 900, 3500)
    e = env(n, .001, .06); e[: int(.02 * SR)] *= (np.arange(int(.02 * SR)) % 300 < 150) + .4
    return x * e
def pluck(m, d=.28):
    n = int(.9 * SR); return lp(saw(hz(m), n, .003), 2600) * env(n, .003, d)
def bass(m, n_s=.22):
    n = int(n_s * SR); t = np.arange(n) / SR
    x = np.sin(2 * np.pi * hz(m) * t) + .25 * lp(saw(hz(m), n), 600)
    return x * env(n, .004, .12)
def pad(root, ivs, n_s, cutoff):
    n = int(n_s * SR)
    x = sum(saw(hz(root + 12 + iv), n, .004) for iv in ivs) / 3
    x = lp(x, cutoff); a = np.minimum(np.arange(n) / (.25 * SR), 1)
    rel = np.minimum((n - np.arange(n)) / (.3 * SR), 1)
    return x * a * rel

# ── arrangement ──
for bar in range(10):  # pads, 0–20s
    t0 = bar * 2.0
    r, ivs = CH[bar % 4]
    cut = 900 if t0 < 3 else 1800
    add(music, pad(r, ivs, 2.05, cut), t0, .16 if t0 >= 3 else .12, rev=.35)

b = 0.0
while b < 20 - 1e-6:
    r, ivs = chord(b); beat_i = int(round(b / BEAT))
    groove = b >= 3.0
    if groove or b < 2.5:          # kick (soft pulse in the hook, full after drop)
        add(music, kick(1.0 if groove else .5), b, .9)
    if groove and beat_i % 2 == 1:
        add(music, clap(), b, .22, rev=.3)
    # 8th-note hats + bass + arp
    for k in (0, 1):
        tt = b + k * BEAT / 2
        if tt < 2.5 or groove:
            add(music, hat(), tt, .07 if k == 0 else .12, pan=.3)
        if groove:
            add(music, bass(r - 12), tt, .5)
            arp = [0, 7, 12, 16 if ivs[1] == 4 else 15][(beat_i * 2 + k) % 4]
            add(music, pluck(r + 12 + arp), tt, .1, pan=-.25 if k else .25, rev=.45)
    b += BEAT

# hook riser into the drop at 3.0
n = int(1.0 * SR); t = np.arange(n) / SR
noise = rng.standard_normal(n); swell = np.zeros(n)
for i in range(0, n, 2048):
    f = 300 + 5000 * (i / n) ** 2
    seg = noise[i:i + 2048]; swell[i:i + len(seg)] = bp(seg, f, f * 1.8)
add(music, swell * (t / 1.0) ** 2, 2.0, .18, rev=.3)

# final hit at 20: kick + A major stab + shimmer tail
add(music, kick(1.2), 20.0, 1.0)
for m in (45, 57, 61, 64, 69, 73):
    n = int(3 * SR)
    add(music, lp(saw(hz(m), n, .004), 2200) * env(n, .004, .9), 20.0, .07, rev=.6)
n = int(2.5 * SR)
add(music, hp(rng.standard_normal(n), 6000) * env(n, .002, .5), 20.0, .05, rev=.5)
add(music, pad(57, [0, 4, 7, 12], 3.0, 1400), 20.0, .14, rev=.5)

# ── SFX (tuned to A major, sit under the music) ──
def whoosh(end, length=.45):
    n = int(length * SR); t = np.arange(n) / SR
    x = np.zeros(n); src = rng.standard_normal(n)
    for i in range(0, n, 1024):
        f = 400 + 3000 * (i / n)
        seg = src[i:i + 1024]; x[i:i + len(seg)] = bp(seg, f, f * 2)
    add(send, x * (t / length) ** 2, end - length, .5)
    add(music, x * (t / length) ** 2, end - length, .07)
for c in (6.0, 10.5, 14.5, 17.0):
    whoosh(c)

def click(g=.25):
    n = int(.03 * SR); return bp(rng.standard_normal(n), 1500, 6000) * env(n, .0005, .004) * g
def blip(m, d=.12):
    n = int(.4 * SR); t = np.arange(n) / SR
    return (np.sin(2 * np.pi * hz(m) * t) + .3 * np.sin(4 * np.pi * hz(m) * t)) * env(n, .002, d)

def shutter(at):                                    # camera shutter: two leaves + tuned body
    add(music, click(.9), at, .55, pan=-.2, rev=.3)
    add(music, click(.7), at + .045, .5, pan=-.2, rev=.3)
    add(music, blip(69, .06), at, .05)

# tether 6–10.5
add(music, click(.6), 7.0, .4, pan=.3)              # USB connects
add(music, blip(81, .1), 7.0, .04, rev=.4)
add(music, click(.5), 7.42, .5, pan=.3)             # tap Start auto-upload
for i, s in enumerate([8.0, 8.5, 9.0]):
    shutter(s)
    add(music, blip([76, 81, 85][i], .1), s + .6, .045, pan=-.3, rev=.45)  # photo lands on the site
    whoosh(s + .55, .35)
# selfie 10.5–14.5
add(music, click(.5), 11.7, .5)
shutter(12.12)
for i, m in enumerate([81, 85, 88, 93, 88, 93]):
    add(music, blip(m), 13.05 + i * .09, .05, pan=(i % 3 - 1) * .4, rev=.4)
# bib 14.5–17
for k in range(4):
    add(music, click(.4), 14.85 + k * .12, .4, pan=.2)
add(music, click(.5), 15.48, .5)
for i, m in enumerate([81, 85, 88, 93]):
    add(music, blip(m), 15.65 + i * .08, .05, pan=(i - 1.5) * .25, rev=.4)
# buy + download 17–20
add(music, click(.5), 17.55, .5)
add(music, blip(76, .15), 17.75, .035, rev=.4)       # QR appears
for i, m in enumerate([69, 73, 76, 81]):             # "All yours." chime
    add(music, blip(m, .3), 18.72 + i * .06, .05, rev=.5)
add(music, click(.5), 19.08, .5)
add(music, blip(88, .2), 19.22, .04, rev=.5)          # saved

# ── reverb send, mix, master ──
ir_n = int(1.6 * SR)
ir = rng.standard_normal((ir_n, 2)) * np.exp(-np.arange(ir_n) / (0.35 * SR))[:, None]
ir = lp(ir.T, 5000).T; ir /= np.abs(ir).sum(axis=0) ** .5 * 8
wet = np.stack([fftconvolve(send[:, c], ir[:, c])[:N] for c in range(2)], axis=1)
mix = hp((music + wet).T, 30).T
mix *= 1 - np.clip((np.arange(N) / SR - 22.2) / .8, 0, 1)[:, None]  # tail fade
mix = np.tanh(mix * 1.4) / np.tanh(1.4)
mix *= 0.89 / np.abs(mix).max()
wavfile.write("audio.wav", SR, (mix * 32767).astype(np.int16))
print("peak ok, seconds:", N / SR)
