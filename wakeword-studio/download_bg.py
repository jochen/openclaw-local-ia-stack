#!/usr/bin/env python3
"""Hintergrund-Audio-Downloads für openwakeword-Training (Stages: rirs, audioset, fma).

Repliziert die Download-Zellen aus notebooks/automatic_model_training.ipynb.
"""
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import datasets
from tqdm import tqdm


def rirs():
    out = "./mit_rirs"
    os.makedirs(out, exist_ok=True)
    if len(os.listdir(out)) > 200:
        print("mit_rirs schon vorhanden")
        return
    ds = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train", streaming=True)
    for row in tqdm(ds):
        name = row["audio"]["path"].split("/")[-1]
        scipy.io.wavfile.write(
            os.path.join(out, name), 16000,
            (row["audio"]["array"] * 32767).astype(np.int16))


def audioset():
    out = "./audioset_16k"
    os.makedirs(out, exist_ok=True)
    files = [str(i) for i in Path("audioset/audio").glob("**/*.flac")]
    ds = datasets.Dataset.from_dict({"audio": files})
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))
    for row in tqdm(ds):
        name = row["audio"]["path"].split("/")[-1].replace(".flac", ".wav")
        scipy.io.wavfile.write(
            os.path.join(out, name), 16000,
            (row["audio"]["array"] * 32767).astype(np.int16))


def fma():
    out = "./fma"
    os.makedirs(out, exist_ok=True)
    n_hours = 2
    target = n_hours * 3600 // 30
    if len(os.listdir(out)) >= target:
        print("fma schon vorhanden")
        return
    ds = datasets.load_dataset("rudraml/fma", name="small",
                               split="train", streaming=True)
    ds = iter(ds.cast_column("audio", datasets.Audio(sampling_rate=16000)))
    for _ in tqdm(range(target)):
        row = next(ds)
        name = row["audio"]["path"].split("/")[-1].replace(".mp3", ".wav")
        scipy.io.wavfile.write(
            os.path.join(out, name), 16000,
            (row["audio"]["array"] * 32767).astype(np.int16))


if __name__ == "__main__":
    {"rirs": rirs, "audioset": audioset, "fma": fma}[sys.argv[1]]()
