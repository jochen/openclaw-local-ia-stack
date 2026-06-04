import io
import re
import logging
from typing import Optional

import numpy as np
import httpx
import librosa
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-analysis")

SPEACHES_URL = "http://speaches:8000"
SER_URL = "http://ser:8002/ser"
STT_MODEL = "guillaumekln/faster-whisper-medium"

app = FastAPI(title="voice-analysis")


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Text-Normalisierung ──────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase + Satzzeichen entfernen."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── WER / CER via Levenshtein ────────────────────────────────────────────────

def levenshtein(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_wer_cer(intended: str, observed: str) -> dict:
    i_norm = normalize_text(intended)
    o_norm = normalize_text(observed)
    i_words = i_norm.split()
    o_words = o_norm.split()
    i_chars = list(i_norm.replace(" ", ""))
    o_chars = list(o_norm.replace(" ", ""))
    wer = levenshtein(i_words, o_words) / max(len(i_words), 1)
    cer = levenshtein(i_chars, o_chars) / max(len(i_chars), 1)
    return {"wer": round(wer, 4), "cer": round(cer, 4), "match": wer < 0.15}


# ── Timing aus Timestamps ────────────────────────────────────────────────────

def compute_timing(words: list, segments: list) -> dict:
    """
    Bevorzugt Top-Level word-Timestamps (wenn via timestamp_granularities=word
    angefordert). Fällt auf Segment-Timestamps zurück, wenn keine Wörter da.
    """
    if words:
        source = "word"
        tokens = words
    elif segments:
        source = "segment"
        tokens = segments
    else:
        return {
            "duration_s": 0.0,
            "word_count": 0,
            "words_per_sec": 0.0,
            "pauses": [],
            "pause_total_s": 0.0,
            "timestamp_source": "none",
        }

    duration_s = tokens[-1].get("end", 0.0)
    word_count = len(tokens)
    words_per_sec = round(word_count / duration_s, 3) if duration_s > 0 else 0.0

    pauses = []
    for i in range(1, len(tokens)):
        gap = tokens[i].get("start", 0) - tokens[i - 1].get("end", 0)
        if gap > 0.3:
            label = tokens[i - 1].get("word", tokens[i - 1].get("text", "")).strip()
            pauses.append({"after_word": label, "gap_s": round(gap, 3)})
    pause_total_s = round(sum(p["gap_s"] for p in pauses), 3)

    return {
        "duration_s": round(duration_s, 3),
        "word_count": word_count,
        "words_per_sec": words_per_sec,
        "pauses": pauses,
        "pause_total_s": pause_total_s,
        "timestamp_source": source,
    }


# ── Prosodie (DSP) ───────────────────────────────────────────────────────────

def compute_prosody(audio_bytes: bytes) -> dict:
    """F0 + RMS aus rohem WAV-Bytes."""
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz
            fmax=librosa.note_to_hz("C7"),   # ~2093 Hz
            sr=sr,
        )
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
        if len(voiced_f0) > 0:
            f0_mean = float(np.mean(voiced_f0))
            f0_std  = float(np.std(voiced_f0))
            f0_min  = float(np.min(voiced_f0))
            f0_max  = float(np.max(voiced_f0))
        else:
            f0_mean = f0_std = f0_min = f0_max = 0.0

        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms))
        rms_std  = float(np.std(rms))

        return {
            "f0_mean_hz": round(f0_mean, 2),
            "f0_std_hz":  round(f0_std,  2),
            "f0_min_hz":  round(f0_min,  2),
            "f0_max_hz":  round(f0_max,  2),
            "rms_mean":   round(rms_mean, 6),
            "rms_std":    round(rms_std,  6),
        }
    except Exception as e:
        logger.warning(f"Prosodie-Fehler: {e}")
        return {
            "f0_mean_hz": 0.0, "f0_std_hz": 0.0,
            "f0_min_hz":  0.0, "f0_max_hz": 0.0,
            "rms_mean":   0.0, "rms_std":   0.0,
        }


# ── Mood-Proxy-Heuristik (Fallback ohne SER) ─────────────────────────────────

def compute_mood_proxy(timing: dict, prosody: dict) -> dict:
    """
    Grobe Heuristik (kein echtes SER!).
    Referenz fuer maennliche Sprachsignale:
      Normal: ~3-4 w/s, Pitch ~90-150 Hz, RMS relativ zum Aufnahme-Pegel.
    """
    wps = timing.get("words_per_sec", 0.0)
    f0  = prosody.get("f0_mean_hz", 0.0)
    rms = prosody.get("rms_mean", 0.0)

    tempo_high  = wps > 4.5
    tempo_low   = 0 < wps < 2.5
    pitch_high  = f0 > 160
    pitch_low   = 0 < f0 < 90
    energy_high = rms > 0.08
    energy_low  = 0 < rms < 0.02

    if pitch_high and tempo_high and energy_high:
        label = "aufgeregt/genervt"
        hint  = ("Hoher Pitch, schnelles Tempo und hohe Energie deuten auf "
                 "Aufregung oder Ungeduld hin.")
    elif pitch_low and tempo_low and energy_low:
        label = "mude/traurig"
        hint  = ("Tiefer Pitch, langsames Tempo und niedrige Energie koennen "
                 "auf Erschoepfung oder gedruckte Stimmung hinweisen.")
    elif pitch_high and energy_high:
        label = "aufgeregt/genervt"
        hint  = ("Erhoehter Pitch und Energie — leicht erhoehte emotionale "
                 "Aktivierung moeglich, Tempo unauffaellig.")
    elif tempo_low and energy_low:
        label = "mude/traurig"
        hint  = ("Langsames Sprechen und geringe Energie — moegliche "
                 "Erschoepfung, Pitch unauffaellig.")
    else:
        label = "neutral"
        hint  = ("Keine auffaelligen Muster in Tempo, Pitch oder Energie — "
                 "Stimmung erscheint unauffaellig neutral.")

    return {
        "label": label,
        "hint":  f"[Heuristik-Fallback, kein echtes SER] {hint}",
    }


# ── /analyze ────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    intended: str = Form(...),
    language: str = Form("de"),
):
    audio_bytes = await file.read()

    # 1. Re-STT via Speaches (mit Wort-Timestamps)
    observed = None
    timing   = None
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{SPEACHES_URL}/v1/audio/transcriptions",
                files={"file": (file.filename or "audio.wav", audio_bytes, "audio/wav")},
                data={
                    "model":           STT_MODEL,
                    "language":        language,
                    "response_format": "verbose_json",
                    # Wort- und Segment-Granularitaet anfordern
                    "timestamp_granularities[]": ["word", "segment"],
                },
            )
        resp.raise_for_status()
        stt_data = resp.json()
        observed = stt_data.get("text", "").strip()
        words    = stt_data.get("words") or []
        segments = stt_data.get("segments") or []
        timing   = compute_timing(words, segments)
        logger.info(f"STT ok: '{observed}' (words={len(words)}, segs={len(segments)})")
    except Exception as e:
        logger.error(f"STT fehlgeschlagen: {e}")
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
            dur = round(len(y) / sr, 3)
        except Exception:
            dur = 0.0
        timing = {
            "duration_s": dur,
            "word_count": 0,
            "words_per_sec": 0.0,
            "pauses": [],
            "pause_total_s": 0.0,
            "timestamp_source": "none",
        }

    # 2. Texttreue
    if observed is not None:
        fidelity = compute_wer_cer(intended, observed)
    else:
        fidelity = {"wer": None, "cer": None, "match": False}

    # 3. Prosodie
    prosody = compute_prosody(audio_bytes)

    # 4. Mood-Proxy (Heuristik, /analyze nutzt noch kein SER)
    mood = compute_mood_proxy(timing, prosody)

    return JSONResponse({
        "intended":      intended,
        "observed":      observed,
        "text_fidelity": fidelity,
        "timing":        timing,
        "prosody":       prosody,
        "mood_proxy":    mood,
    })


# ── /mood ────────────────────────────────────────────────────────────────────

@app.post("/mood")
async def mood_endpoint(file: UploadFile = File(...)):
    """Stimmungsanalyse aus reinem WAV — SER-backed, Prosodie-Fallback."""
    audio_bytes = await file.read()
    if not audio_bytes:
        return JSONResponse(status_code=400, content={"error": "Leere Datei."})

    # Prosodie immer berechnen (bleibt im Response + dient als Fallback)
    try:
        prosody = compute_prosody(audio_bytes)
    except Exception as e:
        logger.error(f"/mood Prosodie-Fehler: {e}")
        return JSONResponse(status_code=422, content={"error": f"WAV konnte nicht verarbeitet werden: {e}"})

    # SER-Anfrage an ser-Dienst
    ser_data = None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                SER_URL,
                files={"file": (file.filename or "audio.wav", audio_bytes, "audio/wav")},
            )
        resp.raise_for_status()
        ser_data = resp.json()
        logger.info(
            f"SER ok: arousal={ser_data.get('arousal')}, "
            f"valence={ser_data.get('valence')}, "
            f"dominance={ser_data.get('dominance')}, "
            f"label={ser_data.get('label')}, "
            f"infer_ms={ser_data.get('infer_ms')}"
        )
    except Exception as e:
        logger.warning(f"SER nicht erreichbar, falle auf Heuristik zurück: {e}")

    if ser_data is not None:
        # SER-Ergebnis verwenden
        arousal   = ser_data.get("arousal", 0.5)
        valence   = ser_data.get("valence", 0.5)
        dominance = ser_data.get("dominance", 0.5)
        label     = ser_data.get("label", "neutral")
        mood_proxy = {
            "label": label,
            "hint":  (
                f"[SER: wav2vec2-large] "
                f"arousal={arousal:.3f}, valence={valence:.3f}, dominance={dominance:.3f}"
            ),
        }
        response = {
            "prosody":    prosody,
            "mood_proxy": mood_proxy,
            "ser": {
                "arousal":   arousal,
                "valence":   valence,
                "dominance": dominance,
                "label":     label,
                "infer_ms":  ser_data.get("infer_ms"),
                "device":    ser_data.get("device"),
            },
        }
    else:
        # Fallback: Prosodie-Heuristik
        dummy_timing = {"words_per_sec": 0.0}
        mood_proxy = compute_mood_proxy(dummy_timing, prosody)
        response = {
            "prosody":    prosody,
            "mood_proxy": mood_proxy,
            "ser":        None,
        }

    return JSONResponse(response)
