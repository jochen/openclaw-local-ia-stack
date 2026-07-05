#!/usr/bin/env python3
"""Score-Diagnose auf rohen (unaugmentierten) Test-Clips, je Aussprache/Stimme."""
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf

os.chdir(os.path.expanduser("~/wakeword-studio"))
sys.path.insert(0, "openwakeword")
from openwakeword.model import Model

m = Model(wakeword_models=["train_out/gaston.onnx"], inference_framework="onnx")

def max_score(path):
    audio, sr = sf.read(path, dtype="int16")
    m.reset()
    best = 0.0
    for i in range(0, len(audio) - 1280, 1280):
        r = m.predict(audio[i:i+1280])
        best = max(best, list(r.values())[0])
    # Rest + Stille anhängen, damit das Fenster das Wortende sieht
    tail = np.zeros(1280 * 8, dtype=np.int16)
    buf = np.concatenate([audio[i+1280:], tail]) if len(audio) > 1280 else tail
    for i in range(0, len(buf) - 1280, 1280):
        r = m.predict(buf[i:i+1280])
        best = max(best, list(r.values())[0])
    return best

rng = np.random.default_rng(0)
files = glob.glob("train_out/gaston/positive_test/*.wav")
groups = defaultdict(list)
for f in files:
    b = os.path.basename(f)
    kind = b.split("_")[0]           # nasal | gastong
    voice = b.split("_")[1]          # Stimmen-Name
    groups[(kind, voice)].append(f)

print(f"{'Gruppe':<38}{'n':>4} {'Recall@0.5':>10} {'Recall@0.2':>10} {'p50':>7}")
for (kind, voice), fl in sorted(groups.items()):
    sel = rng.choice(fl, size=min(40, len(fl)), replace=False)
    s = np.array([max_score(f) for f in sel])
    print(f"{kind + ' / ' + voice:<38}{len(sel):>4} {(s > 0.5).mean():>10.2f} "
          f"{(s > 0.2).mean():>10.2f} {np.median(s):>7.3f}")
