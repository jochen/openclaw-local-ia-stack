#!/usr/bin/env python3
"""FP/h für Streak-Varianten: strikt 3, strikt 2, und 3 mit 1-Frame-Lücken-Toleranz."""
import numpy as np
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
hours = len(val) * 0.08 / 3600

def count_strict(above, need):
    c = 0; hits = 0
    for a in above:
        if a:
            hits += 1
        else:
            if hits >= need:
                c += 1
            hits = 0
    return c + (1 if hits >= need else 0)

def count_gap(above, need):
    """Streak mit Toleranz für genau 1 Unterbrechungs-Frame."""
    c = 0; hits = 0; gap_used = False; below = 0
    for a in above:
        if a:
            hits += 1; below = 0
        else:
            below += 1
            if hits > 0 and below == 1 and not gap_used:
                gap_used = True
                continue
            if hits >= need:
                c += 1
            hits = 0; gap_used = False
    return c + (1 if hits >= need else 0)

print(f"{'Thr':>5} {'3 strikt':>9} {'2 strikt':>9} {'3 mit 1-Gap':>12} {'2 mit 1-Gap':>12}   (FP/h auf %.1f h)" % hours)
for thr in (0.2, 0.25, 0.3, 0.35, 0.4):
    above = s > thr
    print(f"{thr:>5} {count_strict(above,3)/hours:>9.2f} "
          f"{count_strict(above,2)/hours:>9.2f} {count_gap(above,3)/hours:>12.2f} "
          f"{count_gap(above,2)/hours:>12.2f}")
