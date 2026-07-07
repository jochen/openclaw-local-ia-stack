#!/usr/bin/env python3
"""FP/h für die Live-Trigger-Semantik (min_hits=2, 1-Frame-Gap) PLUS
Peak-Bedingung: der beste Score im Streak muss >= min_peak sein.

Motivation: FP vom 2026-07-07 hatte Streak-Scores 0.41/0.68, echte
Gaston-Rufe erreichen 0.94-0.99. Sweep über min_peak-Kandidaten.
Scores werden gecacht (validation_scores.npy) — Varianten sind danach billig.
"""
import os

import numpy as np

CACHE = "validation_scores.npy"

if os.path.exists(CACHE):
    s = np.load(CACHE)
else:
    import onnxruntime as ort

    so = ort.SessionOptions(); so.intra_op_num_threads = 10
    sess = ort.InferenceSession("train_out/gaston.onnx", so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    val = np.load("validation_set_features.npy").astype(np.float32)
    win = np.ascontiguousarray(
        np.lib.stride_tricks.sliding_window_view(val, 16, axis=0).transpose(0, 2, 1))
    s = np.empty(len(win), dtype=np.float32)
    buf = np.empty((1, 16, 96), dtype=np.float32)
    for i in range(len(win)):
        buf[0] = win[i]
        s[i] = sess.run(None, {iname: buf})[0][0, 0]
    np.save(CACHE, s)

hours = len(s) * 0.08 / 3600


def count_gap_peak(scores, thr, need, min_peak):
    """Streaks über thr mit 1-Frame-Gap-Toleranz (wie assistant.py);
    Trigger nur wenn Streak >= need UND Streak-Peak >= min_peak."""
    c = 0; hits = 0; gap_used = False; peak = 0.0
    for x in scores:
        if x > thr:
            hits += 1
            peak = max(peak, x)
        else:
            if hits > 0 and not gap_used:
                gap_used = True
                continue
            if hits >= need and peak >= min_peak:
                c += 1
            hits = 0; gap_used = False; peak = 0.0
    return c + (1 if hits >= need and peak >= min_peak else 0)


peaks = (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)
print(f"Validierung: {hours:.1f} h — Threshold 0.35, min_hits 2, 1-Frame-Gap")
print(f"{'min_peak':>9} {'FP':>5} {'FP/h':>7}")
for p in peaks:
    n = count_gap_peak(s, 0.35, 2, p)
    print(f"{p:>9} {n:>5} {n / hours:>7.2f}")

print()
print("Zum Vergleich min_hits 3 (alte Einstellung):")
for p in (0.0, 0.8):
    n = count_gap_peak(s, 0.35, 3, p)
    print(f"  min_peak {p}: {n} FP = {n / hours:.2f} FP/h")
