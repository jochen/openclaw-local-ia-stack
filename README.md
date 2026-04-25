# openclaw-local-ai-stack

Local AI stack for [OpenClaw](https://openclaw.ai) with persistent memory via [Honcho](https://github.com/plastic-labs/honcho). Runs entirely on-premise using Podman Compose.

## Hardware

| Component | Spec |
|-----------|------|
| GPU | 2× NVIDIA RTX 5060 Ti (Blackwell GB206) |
| VRAM | 16 GB each = **32 GB total** |
| GPU Interconnect | PCIe 4.0 (no NVLink) |
| Host IP | 192.168.111.126 |

> **Note on GPU architecture:** The RTX 5060 Ti is Blackwell (sm100). Some container images require CUDA 12.9+ (`cu129` tags). Standard `latest` tags may not work.

## Architecture

```
Clients (OpenClaw, coding tools, etc.)
         │
         ├─ :11434  vLLM        — LLM inference, pipeline parallel across both GPUs
         ├─ :11435  infinity-emb — Embedding model (nomic-embed-text-v1.5, 768 dims)
         ├─ :8000   speaches    — Speech AI (STT/TTS), GPU with most free VRAM
         └─ :9000   honcho-api  — Memory & social cognition layer
                        ├── honcho-deriver  (background memory worker)
                        ├── PostgreSQL + pgvector
                        └── Redis
```

**One model at a time:** With 32 GB VRAM and an ~18 GB LLM, only one large model can be loaded. Both `current` and `vision` API aliases always point to whatever model is currently loaded. Switch models with `scripts/switch-model.sh`.

## Services

| Service | Image | Port | Notes |
|---------|-------|------|-------|
| `llm` | `vllm/vllm-openai:cu129-nightly` | 11434 | OpenAI-compatible, pipeline-parallel |
| `embeddings` | `michaelf34/infinity:latest` | 11435 | nomic-embed-text-v1.5, 768 dims |
| `speaches` | `speaches-ai/speaches:latest-cuda` | 8000 | Dynamic GPU selection at startup |
| `honcho-api` | local build (honcho-fork) | 9000 | Memory API |
| `honcho-deriver` | local build (honcho-fork) | — | Background worker |
| `database` | `pgvector/pgvector:pg15` | 5432 | localhost only |
| `redis` | `redis:8.2` | 6379 | localhost only |

## Prerequisites

- Podman + podman-compose (`~/.local/bin/podman-compose`)
- NVIDIA Container Toolkit with CDI configured (`nvidia.com/gpu=all`)
- CUDA 12.9 driver (required for RTX 5060 Ti / Blackwell)
- `honcho-fork` repository cloned at `../honcho-fork` (sibling directory)

## Setup

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env: set HF_CACHE to your HuggingFace cache directory
```

Create `../honcho-fork/.env` from the template in that repo.

### 2. Start infrastructure first

```bash
podman-compose up -d database redis embeddings
```

### 3. Start the LLM (downloads model on first run)

```bash
podman-compose up -d llm
# First start downloads ~18 GB from HuggingFace — takes a while
# Monitor: podman logs -f llm
```

### 4. Start remaining services

```bash
podman-compose up -d speaches honcho-api honcho-deriver
```

## Switching models

```bash
# Switch to a different model (restarts vLLM)
scripts/switch-model.sh cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit
scripts/switch-model.sh google/gemma-4-27b-it

# Show current model
scripts/switch-model.sh
```

Models are downloaded from HuggingFace and cached in `$HF_CACHE`. AWQ/GPTQ quantization is auto-detected from the model name.

## Client configuration (OpenClaw, coding tools)

| Setting | Value |
|---------|-------|
| Base URL | `http://192.168.111.126:11434/v1` |
| API Key | `vllm` (any non-empty string) |
| Chat model | `current` |
| Vision model | `vision` |
| Speech (speaches) | `http://192.168.111.126:8000` |
| Honcho API | `http://192.168.111.126:9000` |

Both `current` and `vision` resolve to whatever model is currently loaded.

## Stack management

```bash
# Start all
podman-compose up -d

# Stop all
podman-compose down

# Logs
podman logs -f llm
podman logs -f honcho-api

# Rebuild honcho after code changes
podman-compose build --no-cache honcho-api honcho-deriver
podman stop honcho-api honcho-deriver && podman rm honcho-api honcho-deriver
podman-compose up -d honcho-api honcho-deriver
```

## VRAM budget

```
~18 GB  vLLM  (Qwen3.5-35B-A3B-AWQ-4bit)
~0.3 GB speaches
~0.0 GB infinity-emb (CPU)
────────────────────────────────────────
~18.3 GB / 32 GB used when model is loaded
```

## Related repositories

- [honcho-fork](https://github.com/jochen/honcho) — Local fork of plastic-labs/honcho with dimension patches
- [plastic-labs/honcho](https://github.com/plastic-labs/honcho) — Upstream honcho
