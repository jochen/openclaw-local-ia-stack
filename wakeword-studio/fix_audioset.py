#!/usr/bin/env python3
"""AudioSet-Hintergrund-Audio aus den neuen Parquet-Dateien extrahieren.

Das HF-Repo agkphysics/AudioSet hat die früheren .tar-Archive durch Parquet
(Audio als eingebettete FLAC-Bytes) ersetzt. Lädt 4 Parts (~2.6 GB, ~20 h)
und schreibt 16k-Mono-WAVs nach ./audioset_16k.
"""
import io
import multiprocessing as mp
import os
import subprocess

import numpy as np
import pyarrow.parquet as pq
import scipy.signal
import soundfile as sf

HOME = os.path.expanduser("~/wakeword-studio")
OUT = os.path.join(HOME, "audioset_16k")
PARTS = ["00", "01", "02", "03"]


def convert_row(args):
    name, blob = args
    try:
        data, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if sr != 16000:
            g = np.gcd(sr, 16000)
            mono = scipy.signal.resample_poly(mono, 16000 // g, sr // g)
        sf.write(os.path.join(OUT, name + ".wav"),
                 (np.clip(mono, -1, 1) * 32767).astype(np.int16), 16000,
                 subtype="PCM_16")
        return 0
    except Exception as e:
        print("SKIP", name, type(e).__name__, str(e)[:60], flush=True)
        return 1


def main():
    os.makedirs(OUT, exist_ok=True)
    total = 0
    for part in PARTS:
        dest = os.path.join(HOME, "audioset", f"bal_train_{part}.parquet")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not (os.path.exists(dest) and os.path.getsize(dest) > 100_000_000):
            print(f"lade {part}.parquet ...", flush=True)
            subprocess.run(
                ["wget", "-q", "--tries=5", "--waitretry=30", "-O", dest,
                 "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/"
                 f"data/bal_train/{part}.parquet"], check=True)
        table = pq.read_table(dest, columns=["video_id", "audio"])
        rows = [(vid.as_py(), audio["bytes"].as_py())
                for vid, audio in zip(table["video_id"], table["audio"])]
        print(f"{part}.parquet: {len(rows)} Clips", flush=True)
        with mp.Pool(8) as pool:
            errs = sum(pool.imap_unordered(convert_row, rows, chunksize=8))
        total += len(rows) - errs
        os.remove(dest)  # Parquet nach Extraktion löschen (Platz)
    print(f"FERTIG: {total} WAVs in {OUT}", flush=True)


if __name__ == "__main__":
    main()
