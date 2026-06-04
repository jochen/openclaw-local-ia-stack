# openclaw-local-ai-stack

Lokaler AI-Stack für [OpenClaw](https://openclaw.ai) — läuft vollständig on-premise
über Podman Compose auf einem einzelnen Dual-GPU-Host.

Drei Dienste laufen dauerhaft: **LLM** (llama.cpp), **Embeddings** (infinity-emb)
und **Speech** (speaches: STT/TTS/Diarization). Alle sprechen OpenAI-kompatible APIs.

> **Historie:** Frühere Iterationen nutzten Honcho (Memory-Layer) mit PostgreSQL +
> Redis sowie vLLM/lms statt llama.cpp. Beides ist nicht mehr aktiv — die Honcho-
> Blöcke stehen noch auskommentiert in `compose.yml`, falls sie reaktiviert werden.
> Begründungen zur Modell-/Server-Wahl: `DECISIONS.md`.

## Hardware

| Komponente | Spec |
|-----------|------|
| GPU | 2× NVIDIA RTX 5060 Ti (Blackwell GB206, sm120) |
| VRAM | je 16 GB = **32 GB gesamt** |
| GPU-Interconnect | PCIe 4.0 (kein NVLink) |
| Host-IP | 192.168.111.126 |
| SSH | `user@192.168.111.126` |

> **GPU-Architektur:** Die RTX 5060 Ti ist Blackwell (sm120, CC 12.0). Nur Images
> mit CUDA 12.9+ oder enthaltenem PTX (JIT) funktionieren. Treiber 580+ nötig.

## Architektur

```
Clients (OpenClaw, Voice-Assistant, Coding-Tools, …)
         │
         ├─ :11434  llm          — LLM-Inferenz (llama-server), layer-split über beide GPUs
         ├─ :11435  embeddings   — nomic-embed-text-v1.5 (768 dims), CPU
         ├─ :8000   speaches     — Speech: STT / TTS / Diarization, auf der GPU mit meiste freier VRAM
         └─ :8001   voice-analysis — TTS-/Sprach-Analyse (CPU-only, ruft speaches intern)
```

**Ein großes Modell zur Zeit:** Bei 32 GB VRAM und einem ~24 GB-LLM passt nur ein
großes Modell gleichzeitig. Der API-Alias `current` zeigt immer auf das geladene
Modell. Wechsel via `scripts/switch-model.sh` (recreate des llm-Containers, ~30 s).

## Dienste

| Dienst | Image | Port | Notes |
|---------|-------|------|-------|
| `llm` | `ghcr.io/ggml-org/llama.cpp:server-cuda` | 11434 | OpenAI-kompatibel, layer-split (`--split-mode layer`), `--parallel 2`, Flash-Attention, multimodal via `--mmproj` |
| `embeddings` | `michaelf34/infinity:latest` | 11435 | nomic-embed-text-v1.5, 768 dims, CPU-only, `--url-prefix=/v1` |
| `speaches` | `ghcr.io/speaches-ai/speaches:0.9.0-rc.3-cuda` | 8000 | STT (faster-whisper/ctranslate2, int8 CUDA), TTS (Piper/ONNX), Diarization; dynamische GPU-Wahl beim Start |
| `speaches-warmup` | `alpine` | — | One-shot: lädt STT/TTS-Modelle wenn speaches healthy ist, beendet sich danach |
| `voice-analysis` | `./voice-analysis` (Build) | 8001 | CPU-only; Re-STT + Texttreue (WER/CER) + Timing (Wort-Timestamps) + Prosodie (librosa pyin) + Stimmungs-Heuristik; ruft `speaches:8000` intern |

Honcho-, PostgreSQL- und Redis-Dienste sind in `compose.yml` auskommentiert (s. Historie oben).

## VRAM-Budget

```
~22 GB  llm (Qwen3.5-35B-A3B-Q4_K_M, layer-split auf beide GPUs, LLAMA_CTX=65536)
          GPU0: ~12.3 GB   GPU1: ~10.5 GB
~0.3 GB  speaches (Whisper int8 + Piper) — auf der GPU mit meiste freier VRAM
~0.0 GB  embeddings (CPU-only)
──────────────────────────────────────────────────────────────
~22.8 GB / 32 GB belegt  →  ~8.4 GB frei (GPU0 ~3.3 GB, GPU1 ~5.1 GB)
```

> **Der KV-Cache ist die größte Stellschraube.** `LLAMA_CTX` bestimmt die
> KV-Cache-Größe und damit den Löwenanteil der VRAM-Belegung. Das Absenken von
> `262144` auf `65536` hat ~6 GB freigemacht — genug, um auf GPU1 (~5 GB frei)
> einen zusätzlichen Sprach-Analyse-/SER-Dienst zu betreiben. Bei Bedarf weiter
> senken.
>
> **GPU-Wechsel beim Recreate:** Ein `--force-recreate` des llm-Containers kann an
> einem inaktiven `nvidia-persistenced` scheitern (CDI verlangt
> `/run/nvidia-persistenced/socket`). Dann `sudo systemctl start
> nvidia-persistenced` oder im Zweifel den Host neu starten.

## Voraussetzungen

- Podman + podman-compose (`~/.local/bin/podman-compose`)
- NVIDIA Container Toolkit mit CDI (`nvidia.com/gpu=all`)
- NVIDIA-Treiber 580+ (CUDA 13, nötig für RTX 5060 Ti / Blackwell)
- GGUF-Modelle lokal unter `~/.lmstudio/models/`

> Host-`nvidia-smi` kann durch einen Treiber-/Library-Mismatch fehlschlagen
> (NVML version mismatch) — die Container funktionieren trotzdem über CDI.
> VRAM dann im Container abfragen: `podman exec speaches nvidia-smi`.

## Setup

### 1. Environment konfigurieren

```bash
cp .env.example .env
# Pfade prüfen: LMSTUDIO_MODELS, HF_CACHE, LLAMA_MODEL, LLAMA_CTX
```

### 2. Stack starten

```bash
podman compose up -d
# llm lädt das GGUF (~24 GB VRAM, ~30 s) — Monitor: podman logs -f llm
# speaches-warmup lädt danach automatisch STT/TTS und beendet sich
```

Beim allerersten Start dauert der erste STT-Aufruf ~50 s (CUDA-JIT für Blackwell
sm120). Ab dem zweiten Start greift `speaches-cuda-cache` → alle Aufrufe < 0.5 s.

## Modell wechseln

```bash
# Auf ein anderes Modell wechseln (recreate des llm-Containers, ~30 s)
scripts/switch-model.sh /models/lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf

# Aktuelles Modell + verfügbare Modelle anzeigen
scripts/switch-model.sh
```

`switch-model.sh` schreibt `LLAMA_MODEL` in `.env` und recreate-t den llm-Container.
Modelle liegen lokal als GGUF unter `~/.lmstudio/models/`, im Container unter `/models/`.

## Neue LLM-Modelle herunterladen

```bash
pip3 install --user huggingface_hub   # einmalig

# Modelle suchen (vorquantisierte GGUF von lmstudio-community)
hf models list --author lmstudio-community --filter gguf --sort downloads --limit 20 --format quiet

# Datei laden
hf download lmstudio-community/Qwen3.5-27B-GGUF Qwen3.5-27B-Q4_K_M.gguf \
  --local-dir ~/.lmstudio/models/lmstudio-community/Qwen3.5-27B-GGUF

# Direkt aktivieren
scripts/switch-model.sh /models/lmstudio-community/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf
```

**Quantisierungs-Richtwert für 32 GB VRAM:**

| Modellgröße | Quantisierung | VRAM ca. |
|-------------|---------------|----------|
| 7–8B        | Q8_0          | ~9 GB    |
| 14B         | Q6_K          | ~13 GB   |
| 27–35B      | Q4_K_M        | ~20–24 GB|
| 70B         | Q2_K          | ~28 GB   |

(VRAM zzgl. KV-Cache — bei großem `LLAMA_CTX` deutlich mehr.)

## Speech-Modelle (speaches)

STT/TTS-Modelle stehen in `compose.yml` unter `speaches-warmup` (`STT_MODEL`,
`TTS_MODEL`). `speaches-warmup` lädt sie beim Start automatisch über die API —
kein manueller Download nötig.

Aktuell: STT `guillaumekln/faster-whisper-medium`, TTS `speaches-ai/piper-de_DE-thorsten-medium`.

Endpoints (OpenAI-kompatibel, Base `http://192.168.111.126:8000`):
- `POST /v1/audio/transcriptions` — STT
- `POST /v1/audio/speech` — TTS
- `POST /v1/audio/diarization` — Sprecher-Diarization (mit `known_speaker_*[]`-Referenzen)

Zwei persistente Volumes:
- **`speaches-cache`** — HuggingFace-Weights (überlebt Neustarts)
- **`speaches-cuda-cache`** — kompilierte Blackwell-Kernels (verhindert ~50 s JIT-Delay)

GPU-Wahl: `scripts/speaches-entrypoint.sh` setzt vor dem Start `CUDA_VISIBLE_DEVICES`
auf die GPU mit der meisten freien VRAM (per `nvidia-smi`).

```bash
# Welche Modelle sind gecacht?
podman run --rm -v speaches-cache:/cache alpine ls /cache
```

## Client-Konfiguration (OpenClaw, Voice-Assistant, Coding-Tools)

| Setting | Wert |
|---------|------|
| Provider | `openai` (OpenAI-kompatibel, **nicht** `ollama`) |
| Base URL | `http://192.168.111.126:11434/v1` |
| API Key | `llamacpp` (beliebiger nicht-leerer String) |
| Chat-Modell | `current` |
| Vision | über `--mmproj` am geladenen Modell (Qwen3.5-35B-A3B ist multimodal) → gleicher `current`-Endpoint |
| Embeddings | `http://192.168.111.126:11435/v1` (`nomic-embed-text-v1.5`) |
| Speech | `http://192.168.111.126:8000` |

`current` löst auf das aktuell geladene Modell auf (`--alias current`).

## Stack-Verwaltung

```bash
podman compose up -d            # alle Dienste starten
podman compose down             # alle Dienste stoppen
podman ps                       # laufende Container
podman logs -f llm              # Logs
podman exec speaches nvidia-smi # VRAM (Host-nvidia-smi ggf. kaputt)
```

## Hinweise

**Qwen3 Thinking-Tokens:** Qwen3-Reasoning-Modelle erzeugen vor jeder Antwort
interne `<think>…</think>`-Tokens. Für Clients, die das nicht wollen (kostet
`max_tokens`-Budget), lässt es sich pro Request über
`chat_template_kwargs: {"enable_thinking": false}` (via `extra_body`) abschalten —
das setzt den Qwen3-`<|nothink|>`-Token. Der OpenAI-Parameter `reasoning_effort:
"none"` wird von llama.cpp für Qwen3 **nicht** beachtet (Details: `DECISIONS.md`).

**`modelfiles/`** (gitignored) stammt aus der ollama-Ära und wird von llama.cpp
nicht verwendet — toter Ballast, kann ignoriert werden.

## voice-analysis — `/analyze`-Contract

`POST http://192.168.111.126:8001/analyze` (multipart/form-data)

| Feld | Typ | Pflicht | Beschreibung |
|------|-----|---------|--------------|
| `file` | WAV | ja | PCM16 mono, 16 kHz oder 22 kHz |
| `intended` | string | ja | Erwarteter Sprechtext |
| `language` | string | nein | ISO-639-1, default `de` |

Antwort-Felder:

- **`text_fidelity`** — `wer`, `cer`, `match` (WER < 0.15 gilt als Treffer); normalisiert (lowercase, Satzzeichen weg)
- **`timing`** — `duration_s`, `word_count`, `words_per_sec`, `pauses` (Lücken > 0.3 s), `pause_total_s`, `timestamp_source` (`word` | `segment`)
- **`prosody`** — `f0_mean/std/min/max_hz` (stimmhafte Frames via librosa pyin), `rms_mean/std`
- **`mood_proxy`** — `label` aus {neutral, aufgeregt/genervt, müde/traurig}, `hint` (Klartext für LLM); explizit als grobe Heuristik gekennzeichnet, kein echtes SER

Fehlerfall: wenn Re-STT fehlschlägt, `observed: null`, Prosodie wird trotzdem geliefert.


---

## Verwandte Repositories

- [openclaw-voice-assist](https://github.com/jochen/openclaw_voice_assist) — Voice-Pipeline auf dem Pi, nutzt diesen Stack (speaches + llm)
