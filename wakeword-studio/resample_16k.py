#!/usr/bin/env python3
"""Resampled alle Trainings-Clips auf 16 kHz (in-place).

Nachtrag zu gen_clips.py: piper medium/high-Voices liefern 22.05 kHz,
openwakeword-Augmentierung verlangt strikt 16 kHz.
"""
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

BASE = os.path.expanduser("~/wakeword-studio/train_out/gaston")


def fix(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr == 16000:
        return 0
    mono = data.mean(axis=1)
    g = np.gcd(sr, 16000)
    mono = scipy.signal.resample_poly(mono, 16000 // g, sr // g)
    sf.write(path, (np.clip(mono, -1, 1) * 32767).astype(np.int16), 16000,
             subtype="PCM_16")
    return 1


if __name__ == "__main__":
    files = [str(p) for d in ("positive_train", "positive_test",
                              "negative_train", "negative_test")
             for p in Path(BASE, d).glob("*.wav")]
    print(len(files), "Dateien", flush=True)
    with mp.Pool(10) as pool:
        n = sum(pool.imap_unordered(fix, files, chunksize=64))
    print(f"resampled: {n}, schon 16k: {len(files) - n}", flush=True)
