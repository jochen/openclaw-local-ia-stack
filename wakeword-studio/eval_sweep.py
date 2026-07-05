#!/usr/bin/env python3
"""Threshold-Sweep für gaston.onnx: Recall (Test-Positives) vs. FP/h (11h-Validierung)."""
import numpy as np
import onnxruntime as ort

so = ort.SessionOptions()
so.intra_op_num_threads = 10
sess = ort.InferenceSession("train_out/gaston.onnx", so, providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name

def scores(windows):
    out = np.empty(len(windows), dtype=np.float32)
    buf = np.empty((1, 16, 96), dtype=np.float32)
    for i in range(len(windows)):
        buf[0] = windows[i]
        out[i] = sess.run(None, {iname: buf})[0][0, 0]
    return out

pos = np.load("train_out/gaston/positive_features_test.npy").astype(np.float32)
s_pos = scores(pos)
print("pos fertig", flush=True)

val = np.load("validation_set_features.npy").astype(np.float32)
win = np.lib.stride_tricks.sliding_window_view(val, 16, axis=0)  # (N-15, 96, 16)
win = np.ascontiguousarray(win.transpose(0, 2, 1))               # (N-15, 16, 96)
s_val = scores(win)

hours = len(val) * 0.08 / 3600
print(f"Validierung: {hours:.1f} h")
print(f"{'Thr':>5} {'Recall':>8} {'FP/h':>8}")
for thr in (0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
    above = s_val > thr
    fps = int((above[1:] & ~above[:-1]).sum() + above[0])
    print(f"{thr:>5} {float((s_pos > thr).mean()):>8.3f} {fps / hours:>8.2f}")
print("Score-Verteilung Positives: p10/50/90 =",
      np.percentile(s_pos, [10, 50, 90]).round(3))
