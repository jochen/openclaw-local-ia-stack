"""
SER – Speech Emotion Recognition
Modell: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
Gibt arousal, valence, dominance (je 0-1) plus ein Label zurück.
"""
import io
import time
import logging
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2PreTrainedModel
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ser")

# ── Label-Schwellen (leicht justierbar) ─────────────────────────────────────
AROUSAL_HIGH = 0.52
AROUSAL_LOW  = 0.40
VALENCE_HIGH = 0.55
VALENCE_LOW  = 0.50

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16_000  # wav2vec2 erwartet 16 kHz

# ── Modell-Klassen von HuggingFace-Modellkarte ───────────────────────────────

class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


# ── Modell laden beim Start ──────────────────────────────────────────────────

logger.info(f"Lade Modell auf {DEVICE} ...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = EmotionModel.from_pretrained(MODEL_NAME)
if DEVICE == "cuda":
    model = model.half()  # fp16
model.to(DEVICE)
model.eval()
logger.info(f"Modell geladen auf {DEVICE}")

app = FastAPI(title="ser")


def map_label(arousal: float, valence: float) -> str:
    if arousal > AROUSAL_HIGH and valence < VALENCE_LOW:
        return "genervt/verärgert"
    if arousal > AROUSAL_HIGH and valence >= VALENCE_HIGH:
        return "aufgeregt/freudig"
    if arousal < AROUSAL_LOW and valence < VALENCE_LOW:
        return "müde/traurig"
    return "neutral"


def load_audio_16k(audio_bytes: bytes) -> np.ndarray:
    """WAV-Bytes → 16 kHz mono float32 numpy-Array."""
    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != TARGET_SR:
        # Einfaches resampling via librosa (optional import)
        try:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        except ImportError:
            # Fallback: scipy
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(TARGET_SR, sr)
            y = resample_poly(y, TARGET_SR // g, sr // g).astype(np.float32)
    return y


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/ser")
async def ser_endpoint(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if not audio_bytes:
        return JSONResponse(status_code=400, content={"error": "Leere Datei"})
    try:
        y = load_audio_16k(audio_bytes)
    except Exception as e:
        logger.error(f"Audio-Ladefehler: {e}")
        return JSONResponse(status_code=422, content={"error": f"Audio konnte nicht geladen werden: {e}"})

    t0 = time.monotonic()
    try:
        inputs = processor(y, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(DEVICE)
        if DEVICE == "cuda":
            input_values = input_values.half()
        with torch.no_grad():
            _, logits = model(input_values)
        arr = logits[0].float().cpu().numpy()
        # Modell-Output: [arousal, dominance, valence]
        arousal    = float(np.clip(arr[0], 0.0, 1.0))
        dominance  = float(np.clip(arr[1], 0.0, 1.0))
        valence    = float(np.clip(arr[2], 0.0, 1.0))
    except Exception as e:
        logger.error(f"Inferenz-Fehler: {e}")
        return JSONResponse(status_code=500, content={"error": f"Inferenz fehlgeschlagen: {e}"})

    infer_ms = int((time.monotonic() - t0) * 1000)
    label = map_label(arousal, valence)

    return JSONResponse({
        "arousal":    round(arousal,   4),
        "valence":    round(valence,   4),
        "dominance":  round(dominance, 4),
        "label":      label,
        "infer_ms":   infer_ms,
        "device":     DEVICE,
    })
