# openclaw-local-ai-stack

Local AI stack for [OpenClaw](https://openclaw.ai) with persistent memory via [Honcho](https://github.com/plastic-labs/honcho). Runs entirely on-premise using Podman Compose.

## Hardware

| Component | Spec |
|-----------|------|
| GPU | 2× NVIDIA RTX 5060 Ti (Blackwell GB206) |
| VRAM | 16 GB each = **32 GB total** |
| GPU Interconnect | PCIe 4.0 (no NVLink) |
| Host IP | 192.168.111.126 |

> **Note on GPU architecture:** The RTX 5060 Ti is Blackwell (sm120, Compute Capability 12.0). Use images built with CUDA 12.9+ or that include PTX for JIT compilation. See `DECISIONS.md` for full evaluation history.

## Architecture

```
Clients (OpenClaw, coding tools, etc.)
         │
         ├─ :11434  llama-server  — LLM inference, pipeline-parallel across both GPUs
         ├─ :11435  infinity-emb  — Embedding model (nomic-embed-text-v1.5, 768 dims)
         ├─ :8000   speaches      — Speech AI (STT/TTS), GPU with most free VRAM
         └─ :9000   honcho-api    — Memory & social cognition layer
                        ├── honcho-deriver  (background memory worker)
                        ├── PostgreSQL + pgvector
                        └── Redis
```

**One model at a time:** With 32 GB VRAM and a ~24 GB loaded LLM, only one large model fits. The `current` API alias always points to the loaded model. Switch models with `scripts/switch-model.sh` (restarts the container, ~30 seconds).

## Services

| Service | Image | Port | Notes |
|---------|-------|------|-------|
| `llm` | `ghcr.io/ggml-org/llama.cpp:server-cuda` | 11434 | OpenAI-compatible, pipeline-parallel (`--split-mode layer`) |
| `embeddings` | `michaelf34/infinity:latest` | 11435 | nomic-embed-text-v1.5, 768 dims, CPU |
| `speaches` | `speaches-ai/speaches:latest-cuda` | 8000 | Dynamic GPU selection at startup, CUDA int8 STT |
| `speaches-warmup` | `alpine` | — | Lädt STT/TTS-Modelle beim Start, beendet sich danach |
| `honcho-api` | local build (honcho-fork) | 9000 | Memory API |
| `honcho-deriver` | local build (honcho-fork) | — | Background worker |
| `database` | `pgvector/pgvector:pg15` | 5432 | localhost only |
| `redis` | `redis:8.2` | 6379 | localhost only |

## Prerequisites

- Podman + podman-compose (`~/.local/bin/podman-compose`)
- NVIDIA Container Toolkit with CDI configured (`nvidia.com/gpu=all`)
- NVIDIA driver 580+ (CUDA 13.0, required for RTX 5060 Ti / Blackwell)
- GGUF models already downloaded to `~/.lmstudio/models/`
- `honcho-fork` repository cloned at `../honcho-fork` (sibling directory)

## Setup

### 1. Configure environment

```bash
cp .env.example .env
# Pfade prüfen: LMSTUDIO_MODELS und LLAMA_MODEL
```

Create `../honcho-fork/.env` from the template in that repo.

### 2. Start infrastructure first

```bash
podman compose up -d database redis embeddings
```

### 3. Start the LLM

```bash
podman compose up -d llm
# Lädt das GGUF-Modell (~24 GB VRAM) — dauert ~30 Sekunden
# Monitor: podman logs -f llm
```

### 4. Start remaining services

```bash
podman compose up -d speaches honcho-api honcho-deriver
```

`speaches-warmup` startet automatisch wenn speaches healthy ist, lädt die konfigurierten STT/TTS-Modelle und beendet sich. Beim ersten Start dauert der erste STT-Aufruf ~50 Sekunden wegen CUDA-JIT-Kompilierung (Blackwell sm120). Ab dem zweiten Start greift der `speaches-cuda-cache` und alle Aufrufe sind unter 0.5s.

## Switching models

```bash
# Auf ein anderes Modell wechseln (startet llm-Container neu, ~30 Sek.)
scripts/switch-model.sh /models/lmstudio-community/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf
scripts/switch-model.sh /models/lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf

# Aktuelles Modell und verfügbare Modelle anzeigen
scripts/switch-model.sh
```

Modelle liegen lokal als GGUF unter `~/.lmstudio/models/` und werden im Container unter `/models/` eingebunden.

## Downloading new LLM models

LLM-Modelle (GGUF) werden von HuggingFace heruntergeladen. Einmalige Installation des CLI-Tools:

```bash
pip3 install --user huggingface_hub
```

Danach:

```bash
# Modelle suchen (z.B. GGUF-Modelle von lmstudio-community, sortiert nach Downloads)
hf models list --author lmstudio-community --filter gguf --sort downloads --limit 20 --format quiet

# Infos zu einem bestimmten Modell-Repo
hf models info lmstudio-community/Qwen3.5-27B-GGUF

# Einzelne GGUF-Datei herunterladen
hf download \
  lmstudio-community/Qwen3.5-27B-GGUF \
  Qwen3.5-27B-Q4_K_M.gguf \
  --local-dir ~/.lmstudio/models/lmstudio-community/Qwen3.5-27B-GGUF

# Anschließend direkt laden
scripts/switch-model.sh /models/lmstudio-community/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf
```

**Quantisierungs-Richtwert für 32 GB VRAM:**

| Modellgröße | Empfohlene Quantisierung | VRAM ca. |
|-------------|--------------------------|----------|
| 7–8B        | Q8_0                     | ~9 GB    |
| 14B         | Q6_K                     | ~13 GB   |
| 27–35B      | Q4_K_M                   | ~20–24 GB|
| 70B         | Q2_K                     | ~28 GB   |

Modellnamen auf HuggingFace: am besten unter `lmstudio-community` suchen — dort gibt es vorquantisierte GGUF-Pakete für die gängigen Modelle.

## Speech models (speaches)

STT- und TTS-Modelle sind in `compose.yml` unter `speaches-warmup` konfiguriert (`STT_MODEL`, `TTS_MODEL`). Beim Start lädt `speaches-warmup` die Modelle automatisch über die speaches-API — kein manueller Download nötig. Die Modelle werden im Podman-Volume `speaches-cache` gecacht.

Zwei persistente Volumes:
- **`speaches-cache`** — HuggingFace Modell-Weights (bleibt über Neustarts erhalten)
- **`speaches-cuda-cache`** — Kompilierte CUDA-Kernels für Blackwell (verhindert ~50s JIT-Delay)

```bash
# Welche Modelle sind bereits gecacht?
podman run --rm -v speaches-cache:/cache alpine ls /cache

# STT/TTS-Modelle manuell ändern: STT_MODEL / TTS_MODEL in compose.yml anpassen
```

## Client configuration (OpenClaw, coding tools)

| Setting | Value |
|---------|-------|
| Provider | `openai` (OpenAI-kompatibel, **nicht** `ollama`) |
| Base URL | `http://192.168.111.126:11434/v1` |
| API Key | `llamacpp` (beliebiger nicht-leerer String) |
| Chat model | `current` |
| Vision model | `vision` *(aktuell nicht konfiguriert)* |
| Speech (speaches) | `http://192.168.111.126:8000` |
| Honcho API | `http://192.168.111.126:9000` |

`current` löst auf das aktuell geladene Modell auf (Alias via `--alias current`).

## Stack management

```bash
# Alle Services starten
podman compose up -d

# Alle Services stoppen
podman compose down

# Logs
podman logs -f llm
podman logs -f honcho-api

# Honcho nach Code-Änderungen neu bauen
podman compose build --no-cache honcho-api honcho-deriver
podman stop honcho-api honcho-deriver && podman rm honcho-api honcho-deriver
podman compose up -d honcho-api honcho-deriver
```

## VRAM budget

```
~24.4 GB  llama-server (Qwen3.5-35B-A3B-Q4_K_M.gguf, aufgeteilt auf 2 GPUs)
           GPU0: ~12.4 GB  GPU1: ~12.0 GB
 ~0.3 GB  speaches
 ~0.0 GB  infinity-emb (CPU-only)
──────────────────────────────────────────────────────
~24.7 GB / 32 GB VRAM genutzt wenn Modell geladen
```

## Honcho + thinking models (Qwen3)

Qwen3 reasoning models generate internal `<think>…</think>` tokens before every response. Without suppression, these consume honcho's entire `max_tokens` budget, leaving empty content — the deriver produces no observations and the summarizer falls back to stub text.

`honcho-fork/.env` disables thinking for deriver and summarizer calls:

```
DERIVER_MODEL_CONFIG__THINKING_EFFORT=none
SUMMARY_MODEL_CONFIG__THINKING_EFFORT=none
```

This translates to `chat_template_kwargs: {"enable_thinking": false}` in the llama.cpp request (via `extra_body`), which inserts the Qwen3 `<|nothink|>` token. The standard `reasoning_effort: "none"` OpenAI parameter is **not** honored by llama.cpp for Qwen3 — see `DECISIONS.md` for the full diagnosis.

**Switching models:**
- **Non-thinking model** (Qwen2.5, Llama 3, Mistral, …): setting is silently ignored by llama.cpp, leave it as-is
- **Another thinking model** (DeepSeek-R1, future Qwen releases): same mechanism works
- **Real OpenAI API** (o1/o3): would require a different approach — not relevant for this local stack

---

## Related repositories

- [honcho-fork](https://github.com/jochen/honcho) — Local fork of plastic-labs/honcho with dimension patches
- [plastic-labs/honcho](https://github.com/plastic-labs/honcho) — Upstream honcho
