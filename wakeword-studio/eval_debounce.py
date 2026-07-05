#!/usr/bin/env python3
"""FP/h bei 3-Frame-Debounce (wie assistant.py) je Threshold."""
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
print(f"{'Thr':>5} {'FP/h roh':>9} {'FP/h 3-Frame-Streak':>20}")
for thr in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
    above = s > thr
    raw = int((above[1:] & ~above[:-1]).sum() + above[0])
    streak = np.convolve(above.astype(int), np.ones(3, dtype=int), "valid") >= 3
    ev = int((streak[1:] & ~streak[:-1]).sum() + streak[0])
    print(f"{thr:>5} {raw/hours:>9.2f} {ev/hours:>20.2f}")
