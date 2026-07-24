# openclaw-local-ai-stack

> ⚠️ **Archiviert (2026-07-24).** Dieses Repo ist in zwei Nachfolge-Repos
> aufgeteilt worden, weil die Dienste inzwischen auf drei getrennten Hosts mit
> unterschiedlicher Rolle laufen:
> - [openclaw-fablab-llm](https://github.com/jochen/openclaw-fablab-llm) — LLM + Embeddings (192.168.111.126)
> - [openclaw-voice-stack](https://github.com/jochen/openclaw-voice-stack) — speaches/ser/voice-analysis (rouven, gastonllm)
>
> Volle Historie/Entscheidungen bis zum Split bleiben hier erhalten, wird
> aber nicht mehr aktiv gepflegt.

Lokaler AI-Stack für [OpenClaw](https://openclaw.ai) — läuft vollständig on-premise
über Podman Compose auf einem einzelnen Dual-GPU-Host.

Vier Dienste laufen dauerhaft: **LLM** (llama.cpp), **Embeddings** (infinity-emb),
**Speech** (speaches: STT/TTS/Diarization) und **SER** (Speech Emotion Recognition,
wav2vec2-large auf GPU). Alle sprechen OpenAI-kompatible oder eigene REST-APIs.

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
         ├─ :11434  llm          — LLM-Inferenz (llama-server, Router-Mode), layer-split über beide GPUs
         ├─ :11435  embeddings   — BAAI/bge-m3 (1024 dims, multilingual), GPU
         ├─ :8000   speaches     — Speech: STT / TTS / Diarization, auf der GPU mit meiste freier VRAM
         ├─ :8001   voice-analysis — Sprach-Analyse (CPU-only); /mood nutzt intern den ser-Dienst
         └─ :8002   ser          — Speech Emotion Recognition (GPU1, wav2vec2-large, fp16)
```

**Ein großes Modell zur Zeit, Wechsel per Request (Router-Mode):** Bei 32 GB VRAM
passt nur ein großes LLM gleichzeitig. Der llm-Container läuft im llama-server
**Router-Mode** (`--models-preset llm-presets.ini`, `--models-max 1`): Der
Modellname im API-Request (`"model": "qwen" | "gemma" | "ornith"`) wählt das
Modell; ist ein anderes geladen, wird es automatisch entladen und das angefragte
mit seinen Preset-Einstellungen geladen (~20 s, LRU). Der Alias `current` zeigt
aus Kompatibilitätsgründen auf `qwen`. Siehe „Weitere Modelle hinzufügen".

## Dienste

| Dienst | Image | Port | Notes |
|---------|-------|------|-------|
| `llm` | `ghcr.io/ggml-org/llama.cpp:server-cuda` | 11434 | OpenAI-kompatibel, **Router-Mode** (Modelle aus `llm-presets.ini`, on-demand-Swap), layer-split, Flash-Attention, KV-Cache q8_0, multimodal via `mmproj`, MTP wo verfügbar |
| `embeddings` | `michaelf34/infinity:latest` | 11435 | BAAI/bge-m3, 1024 dims, multilingual, GPU (~1,6 GB VRAM), `--url-prefix=/v1` |
| `speaches` | `ghcr.io/speaches-ai/speaches:0.9.0-rc.3-cuda` | 8000 | STT (faster-whisper/ctranslate2, int8 CUDA), TTS (Piper/ONNX), Diarization; dynamische GPU-Wahl beim Start |
| `speaches-warmup` | `alpine` | — | One-shot: lädt STT/TTS-Modelle wenn speaches healthy ist, beendet sich danach |
| `ser` | `./ser` (Build) | 8002 | **GPU1** (CUDA_VISIBLE_DEVICES=1, fp16); `POST /ser` (WAV → arousal/valence/dominance + label); Modell: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` (~600 MiB VRAM) |
| `voice-analysis` | `./voice-analysis` (Build) | 8001 | CPU-only; Re-STT + Texttreue (WER/CER) + Timing + Prosodie (librosa pyin); `/mood` SER-backed (ruft `ser:8002` intern, Prosodie-Heuristik als Fallback) |

Honcho-, PostgreSQL- und Redis-Dienste sind in `compose.yml` auskommentiert (s. Historie oben).

## VRAM-Budget

```
~25 GB  llm (variiert je Modell; Beispiel qwen: Q4_K_S + mmproj + MTP,
          ctx 131072 mit q8_0-KV-Cache, layer-split auf beide GPUs)
          GPU0: ~12.4 GB   GPU1: ~12.3 GB
~0.4 GB  speaches (Whisper int8 + Piper) — GPU1
~0.6 GB  ser (wav2vec2-large, fp16) — GPU1 (gepinnt via CUDA_VISIBLE_DEVICES=1)
~1.6 GB  embeddings (bge-m3, GPU1)
──────────────────────────────────────────────────────────────
Regel: ~10 % pro GPU frei lassen (max. ~14.7 GB belegt je GPU).
GPU1 ist wegen der Dauer-Dienste meist der Engpass.
```

> **Der KV-Cache ist die größte Stellschraube.** `ctx-size` (pro Preset-Sektion
> in `llm-presets.ini`) bestimmt die KV-Cache-Größe und damit den Löwenanteil
> der variablen VRAM-Belegung. Mit `cache-type-k/v = q8_0` halbiert sich der
> KV-Bedarf gegenüber f16 (Qualitätsverlust vernachlässigbar; Context-Shift
> funktioniert damit nicht). Achtung: SWA-/Hybrid- und MoE-Modelle skalieren
> ihren KV-Cache nicht linear mit ctx — immer messen statt rechnen.
>
> **`n_ctx` pro Slot = `ctx-size` / `parallel`.** Der KV-Cache-Verbrauch
> hängt nur von `ctx-size` ab, nicht von `parallel` (kv-unified) — bei
> `parallel = 1` bekommt eine einzelne Session den vollen Kontext ohne
> zusätzlichen VRAM-Bedarf, dafür kann nur 1 Request gleichzeitig bedient werden.
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

## Modelle & Wechsel (Router-Mode)

Die Modelle stehen in **`llm-presets.ini`** (INI-Keys = llama-server-CLI-Optionen
ohne führende Bindestriche; `[*]` = globale Defaults; Sektionsname = Modellname
im API-Request). Der Wechsel passiert **automatisch per Request** — kein Skript,
kein Restart:

```bash
curl http://localhost:11434/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "ornith", "messages": [{"role": "user", "content": "Hi"}]}'
# → entlädt das aktuelle Modell, lädt ornith mit seinen Preset-Settings (~20 s)
```

Aktuelles Set, **`full`-Modus** (Stand 2026-07-09, Messwerte siehe
`llm-presets.ini`-Kommentare):

| Name | Modell | ctx | MTP | Speed |
|------|--------|-----|-----|-------|
| `qwen` (Alias `current`) | Qwen3.6-27B heretic Q4_K_S + mmproj | 131072 | nativ (`spec-type`) | ~30–40 t/s |
| `gemma` | gemma-4-31B-it heretic Q4_K_M + mmproj | 131072 | — (Drafter passt bei 131072 nicht mehr) | ~20 t/s |
| `ornith` | Ornith-35B heretic Q4_K_M (MoE A3B) + mmproj | 131072 | — (MoE ist so schnell) | ~113 t/s |

Alle drei sind Reasoning-Modelle (Antwort in `reasoning_content` + `content` —
`max_tokens` großzügig setzen). `/v1/models` listet zusätzlich ein Pseudo-Modell
`default`; ignorieren. Clients: Timeouts ≥ 60 s wegen Swap-Latenz.

### Zwei ctx-Profile: `llm-presets.ini.full` / `.compact`

`llm-presets.ini` ist die **live** Datei, die der `llm`-Container mountet —
llama-server liest sie nur beim Start (kein Live-Reload), ein reiner
Datei-Edit wirkt also erst nach `--force-recreate`. Es gibt zwei fertige
Varianten im Repo, geschaltet über `scripts/voice-stack.sh` (siehe
[Stack-Verwaltung](#stack-verwaltung)):

- **`llm-presets.ini.full`** — ctx 131072 überall, mmproj überall, kein
  `tensor-split`. Braucht fast das komplette VRAM beider GPUs (siehe
  VRAM-Budget) → nur wenn speaches/ser/voice-analysis **gestoppt** sind.
- **`llm-presets.ini.compact`** — ctx 65536, globales `tensor-split = 59,41`
  (verschiebt LLM-Last Richtung GPU0), gemma ohne mmproj (dafür eigenes
  `[gemma-vision]`-Preset bei ctx 65536) → lässt GPU1 genug Reserve für die
  Sprach-Pipeline nebenher.

```bash
scripts/voice-stack.sh stop   # Sprach-Pipeline aus + llm -> full  (131072, mmproj)
scripts/voice-stack.sh start  # llm -> compact (65536, GPU1-Reserve) + Sprach-Pipeline an
scripts/voice-stack.sh status # Container-Status + aktives ctx-Preset + GPU-VRAM
```

Manuell auf ein drittes Preset wechseln (z.B. eigene Datei): Preset-Datei nach
`llm-presets.ini` kopieren, dann `~/.local/bin/podman-compose up -d
--force-recreate llm` (ohne `--force-recreate` ignoriert podman-compose die
Änderung).

**Fallback Single-Model-Modus:** `LLAMA_PRESETS=` in `.env` leeren → der alte
Pfad über `LLAMA_MODEL`/`LLAMA_CTX`/… und `scripts/switch-model.sh` greift wieder.

## Weitere Modelle hinzufügen

Checkliste (so wurden gemma/ornith am 2026-07-03 eingerichtet):

1. **GGUF finden & laden.** `scripts/hf-model.sh search <repo>` zeigt Dateien
   und Größen, `scripts/hf-model.sh download <repo> <datei>` lädt nach
   `~/.lmstudio/models/<repo>/` (im Container `/models/<repo>/`). Quant-Wahl:
   Q4_K_M als Default; Weights-Datei + ~2–8 GB (KV+Compute) muss unter das
   VRAM-Budget passen (s.u.). **Achtung:** ik_llama-Quants (IQ4_KS, IQ2_KT, …)
   lädt das mainline-Image nicht ("invalid ggml type"). mmproj-Datei (Vision)
   aus demselben Repo mitladen, falls vorhanden.
2. **MTP prüfen** (Speed ~1,5–3×, nur bei Dense-Modellen relevant):
   - *Qwen3.6-style (nativ):* GGUF muss „MTP-Preserved" sein (MTP-Tensoren
     enthalten) → nur `spec-type = draft-mtp` setzen.
   - *Gemma-4-style (Drafter):* separates kleines Drafter-GGUF nötig (z. B.
     `unsloth/<modell>-GGUF` Ordner `MTP/`, ~0,5 GB) → `model-draft = <pfad>`
     **und** `spec-type = draft-mtp`. Funktioniert auch mit abliterierten
     Targets (gemessen: ~90 % Acceptance).
   - *MoE-A3B-Modelle:* MTP meist unnötig — wenige aktive Parameter sind auch
     so schnell (ornith: 113 t/s).
   - Nach dem Einrichten `draft_n_accepted/draft_n` in der API-Response prüfen:
     unter ~40 % Acceptance bremst MTP eher → wieder entfernen.
3. **Preset-Sektion in `llm-presets.ini` anlegen.** Muster: `model`, ggf.
   `mmproj`, `ctx-size` (konservativ starten, z. B. 32768), `cache-type-k` /
   `cache-type-v` = `q8_0` (halbiert KV-VRAM; Context-Shift geht damit nicht).
   Globales (ngl, flash-attn, threads, parallel) erbt aus `[*]`.
4. **Aktivieren:** `~/.local/bin/podman-compose up -d --force-recreate llm`
   (ohne `--force-recreate` ignoriert podman-compose die Änderung!).
5. **VRAM vermessen & ctx maximieren.** Modell per Request laden, dann
   `nvidia-smi --query-gpu=index,memory.used --format=csv`. Regel: **~10 %
   pro GPU frei lassen** (max. ~14,7 GB belegt). GPU1 trägt zusätzlich
   dauerhaft bge-m3 (~1,6 GB) + speaches (~0,4 GB). ctx schrittweise erhöhen
   und erneut messen — **nicht rechnen**: SWA-/Hybrid-Modelle (Qwen3.x,
   Gemma) und MoE skalieren ihren KV-Cache nicht linear mit ctx
   (gemessen: qwen +2,1 GB, ornith +0,36 GB pro GPU je +32k).
6. **Funktionstest** über beide Namen (`/v1/models`, Chat-Request) und
   **dokumentieren**: Messwerte als Kommentar in die INI-Sektion; Drawer im
   MemPalace (Wing `user-a520-aorus-elite-home-user-ai-stack`); OpenClaw-Configs
   ergänzen, wenn das Modell dort wählbar sein soll.

## Neue LLM-Modelle suchen & herunterladen

```bash
# .gguf-Dateien eines HF-Repos mit Größe auflisten (optional nach Substring filtern)
scripts/hf-model.sh search llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF
scripts/hf-model.sh search llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF Q4_K

# Datei herunterladen — landet automatisch unter
# ~/.lmstudio/models/<repo>/<datei> (gleiche Struktur wie LM Studio)
scripts/hf-model.sh download llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF \
  Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q4_K_S.gguf

# Aktivieren: Sektion in llm-presets.ini anlegen (siehe „Weitere Modelle
# hinzufügen"), dann: ~/.local/bin/podman-compose up -d --force-recreate llm
```

`hf-model.sh` nutzt die HF-API (`curl`/`jq`) zum Suchen und `hf download`
(huggingface_hub, `pip3 install --user huggingface_hub`) zum Laden.

⚠️ **Quant-Format beachten:** Nur Standard-ggml-Quants (`Q4_K_M`, `Q4_K_S`,
`IQ4_XS`, `IQ4_NL`, …) funktionieren mit dem `ggml-org/llama.cpp`-Image.
Quants wie `IQ4_KS`, `IQ2_KT` etc. (ik_llama.cpp-Erweiterungen, höhere
ggml-Type-IDs) schlagen beim Laden fehl
(`invalid ggml type ... should be in [0, 42)`).

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

**VRAM für llm freimachen:** `scripts/voice-stack.sh` schaltet speaches, ser
und voice-analysis zusammen mit dem llm-ctx-Profil um (embeddings bleibt
davon unberührt, läuft immer durch):

```bash
scripts/voice-stack.sh stop   # Sprach-Pipeline aus, llm -> full (ctx 131072 + mmproj überall)
scripts/voice-stack.sh start  # llm -> compact (ctx 65536, GPU1-Reserve), danach Sprach-Pipeline an
scripts/voice-stack.sh status # Container-Status + aktives ctx-Preset + GPU-VRAM
```

`stop`/`start` recreaten dabei immer den `llm`-Container (auch wenn das
Preset scheinbar schon passt) — llama-server liest `llm-presets.ini` nur beim
eigenen Start, ein reiner Datei-Edit ohne Recreate hätte keine Wirkung.
restart-policy aller Container ist `unless-stopped`: manuell gestoppte
Container bleiben gestoppt, kein Auto-Restart außer bei Host-Reboot.

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

## voice-analysis — `/mood`-Contract

`POST http://192.168.111.126:8001/mood` (multipart/form-data)

Schnelle Stimmungsanalyse von **Nutzer-Audio** — kein STT, kein Textvergleich.
Gedacht für den Voice-Assistant, der das `label` in den LLM-Prompt einspeist.
Das Label kommt jetzt **vom SER-Dienst** (wav2vec2-large, dimensional), nicht mehr
aus der Pitch/Energie-Heuristik. Bei Ausfall von `ser` greift Prosodie-Heuristik als
Fallback (kein Crash, `ser`-Feld ist dann `null`).

| Feld | Typ | Pflicht | Beschreibung |
|------|-----|---------|--------------|
| `file` | WAV | ja | PCM16 mono, beliebige Sample-Rate |

Antwort-Felder:

- **`prosody`** — `f0_mean/std/min/max_hz` (stimmhafte Frames via librosa pyin), `rms_mean/std`
- **`mood_proxy`** — `label` aus {neutral, genervt/verärgert, aufgeregt/freudig, müde/traurig}, `hint` (Rohwerte arousal/valence/dominance oder Fallback-Hinweis)
- **`ser`** — Rohwerte vom SER-Dienst: `arousal`, `valence`, `dominance` (je 0–1), `label`, `infer_ms`, `device`; `null` wenn Dienst nicht erreichbar

Label-Mapping (Schwellen in `ser/app.py` justierbar):

| Bedingung | Label |
|-----------|-------|
| arousal > 0.60 und valence < 0.45 | `genervt/verärgert` |
| arousal > 0.60 und valence ≥ 0.55 | `aufgeregt/freudig` |
| arousal < 0.40 und valence < 0.45 | `müde/traurig` |
| sonst | `neutral` |

Fehlerfall: `422` bei ungültigem/leerem WAV.
