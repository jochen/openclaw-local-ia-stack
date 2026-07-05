#!/usr/bin/env python3
"""Robuste Ersatz-Downloads für RIR + Hintergrund-Audio (ohne datasets/ffmpeg).

Ersetzt die gescheiterten Stufen 4-6 aus setup_downloads.sh:
- RIRs direkt vom MIT (Audio.zip, 32k-WAVs -> 16k)
- AudioSet-Tars (bal_train08+09) via HF, FLAC -> 16k-WAV (soundfile+scipy)
- FMA entfällt (Script-Dataset, von neuen datasets-Versionen nicht mehr
  unterstützt) — AudioSet enthält genug Musik/Noise-Vielfalt.
"""
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

HOME = os.path.expanduser("~/wakeword-studio")


def to_16k(args):
    src, dst = args
    try:
        data, sr = sf.read(src, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if sr != 16000:
            g = np.gcd(sr, 16000)
            mono = scipy.signal.resample_poly(mono, 16000 // g, sr // g)
        sf.write(dst, (np.clip(mono, -1, 1) * 32767).astype(np.int16), 16000,
                 subtype="PCM_16")
        return None
    except Exception as e:  # kaputte Einzeldateien überspringen
        return f"{src}: {e}"


def convert_dir(files, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(str(f), os.path.join(out_dir, Path(f).stem + ".wav")) for f in files]
    errs = 0
    with mp.Pool(8) as pool:
        for r in pool.imap_unordered(to_16k, jobs, chunksize=16):
            if r:
                errs += 1
                print("SKIP", r, flush=True)
    print(f"{out_dir}: {len(jobs) - errs} konvertiert, {errs} übersprungen", flush=True)


def wget(url, out):
    subprocess.run(["wget", "-q", "--tries=5", "--retry-on-http-error=429,503",
                    "--waitretry=30", "-O", out, url], check=True)


def rirs():
    out = os.path.join(HOME, "mit_rirs")
    if os.path.isdir(out) and len(os.listdir(out)) > 200:
        print("mit_rirs vorhanden", flush=True)
        return
    zpath = os.path.join(HOME, "mit_audio.zip")
    if not (os.path.exists(zpath) and os.path.getsize(zpath) > 10_000_000):
        wget("https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip", zpath)
    subprocess.run(["unzip", "-oq", zpath, "-d", os.path.join(HOME, "mit_audio")],
                   check=True)
    files = list(Path(HOME, "mit_audio").glob("**/*.wav"))
    print(f"MIT RIRs: {len(files)} Dateien", flush=True)
    convert_dir(files, out)


def audioset():
    out = os.path.join(HOME, "audioset_16k")
    for part in ("bal_train08", "bal_train09"):
        tar = os.path.join(HOME, "audioset", f"{part}.tar")
        os.makedirs(os.path.dirname(tar), exist_ok=True)
        if not (os.path.exists(tar) and os.path.getsize(tar) > 100_000_000):
            print(f"lade {part}.tar ...", flush=True)
            wget("https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/"
                 f"{part}.tar", tar)
        subprocess.run(["tar", "-xf", tar, "-C", os.path.join(HOME, "audioset")],
                       check=True)
    files = list(Path(HOME, "audioset", "audio").glob("**/*.flac"))
    print(f"AudioSet: {len(files)} FLACs", flush=True)
    convert_dir(files, out)


if __name__ == "__main__":
    stages = sys.argv[1:] or ["rirs", "audioset"]
    for s in stages:
        print(f"===== {s} =====", flush=True)
        {"rirs": rirs, "audioset": audioset}[s]()
    print("FERTIG", flush=True)
