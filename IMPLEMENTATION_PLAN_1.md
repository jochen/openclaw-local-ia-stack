# Implementation Plan 1: lms (LM Studio CLI) in Podman

## Context

After evaluating Ollama (too slow on PCIe multi-GPU) and vLLM (OOM due to eager
KV cache pre-allocation), we're containerizing LM Studio's own llama.cpp inference
engine via its headless CLI (`lms`). See `DECISIONS.md` for full rationale.

**Goal:** Replace the current broken `llm` service in `compose.yml` with a working
container that runs `lms server start` headlessly, serves the OpenAI-compatible API
on port 11434, and supports hot-swap model switching.

**Hardware:** 2× NVIDIA RTX 5060 Ti (Blackwell, 16 GB each), host 192.168.111.126

---

## What we know about the lms binary

- **Location:** `/home/user/.lmstudio/bin/lms`
- **Type:** Native ELF 64-bit, minimal glibc deps (libc, libpthread, libdl, libm)
- **Inference engine:** llama.cpp v2.13.0 with CUDA 12 backend
- **Bundled CUDA libs:** `/home/user/.lmstudio/extensions/backends/vendor/cuda12/`
  (includes libcublas.so.12, libcublasLt.so.12 — no host CUDA Toolkit needed)
- **Models:** `/home/user/.lmstudio/models/`
- **Runtime state:** Written to `~/.lmstudio/.internal/` (needs to be writable)

**Key question to verify first:** Does `lms server start` stay in the foreground
(blocking) or daemonize (exit immediately)? This determines the entrypoint approach.

---

## Files to create / modify

| File | Action |
|------|--------|
| `compose.yml` | Replace vLLM `llm` service with lms-based service |
| `scripts/lms-entrypoint.sh` | New: starts server, loads model, keeps container alive |
| `scripts/switch-model.sh` | Update: use `lms load/unload` for hot-swap |
| `.env` / `.env.example` | Update: `LLAMA_MODEL` path variable |
| `honcho-fork/.env` | Update: LLM URL from `http://llm:8000/v1` → `http://llm:11434/v1` |
| `DECISIONS.md` | Already updated ✅ |

---

## Step 1: Verify lms server behavior

Before building anything, run this on the host to understand the process model:

```bash
/home/user/.lmstudio/bin/lms server start --port 11434 --bind 0.0.0.0 &
LMS_PID=$!
sleep 3
curl -s http://localhost:11434/v1/models
ps aux | grep lms
# Does $LMS_PID still exist? → daemonized if yes, foreground if no (process exited)
/home/user/.lmstudio/bin/lms server stop
```

Also check: what model name does the API use?
```bash
/home/user/.lmstudio/bin/lms load \
  "lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf"
curl -s http://localhost:11434/v1/models | python3 -m json.tool
# Note the `id` field — that's what clients must send as `model:`
```

---

## Step 2: compose.yml — replace llm service

Replace the current (broken) vLLM `llm` service:

```yaml
# ── LLM Server (lms / LM Studio headless) ──────────────────────────────────────
llm:
  image: ubuntu:22.04
  container_name: llm
  devices:
    - nvidia.com/gpu=all
  security_opt:
    - label=disable
  ports:
    - "11434:11434"
  volumes:
    - /home/user/.lmstudio/bin:/home/user/.lmstudio/bin:ro
    - /home/user/.lmstudio/extensions:/home/user/.lmstudio/extensions:ro
    - /home/user/.lmstudio/models:/home/user/.lmstudio/models:ro
    - lms-state:/home/user/.lmstudio/.internal
    - ./scripts/lms-entrypoint.sh:/entrypoint.sh:ro
  environment:
    - HOME=/home/user
    - LLAMA_MODEL=${LLAMA_MODEL}
  entrypoint: ["/bin/bash", "/entrypoint.sh"]
  restart: unless-stopped
```

Remove `vllm-cache` from volumes section, add `lms-state`.

---

## Step 3: scripts/lms-entrypoint.sh

Two variants depending on what Step 1 reveals:

**Variant A: lms server start daemonizes (exits immediately)**
```bash
#!/bin/bash
set -e
LMS=/home/user/.lmstudio/bin/lms

echo "Starting LM Studio server..."
$LMS server start --port 11434 --bind 0.0.0.0

# Wait for API to be ready
until curl -sf http://localhost:11434/v1/models >/dev/null 2>&1; do
    sleep 2
done
echo "Server ready."

# Load initial model
if [ -n "$LLAMA_MODEL" ]; then
    echo "Loading model: $LLAMA_MODEL"
    $LMS load "$LLAMA_MODEL"
fi

# Keep container alive
exec sleep infinity
```

**Variant B: lms server start blocks (foreground)**
```bash
#!/bin/bash
LMS=/home/user/.lmstudio/bin/lms

# Load model in background after server starts
(
    until curl -sf http://localhost:11434/v1/models >/dev/null 2>&1; do
        sleep 2
    done
    [ -n "$LLAMA_MODEL" ] && $LMS load "$LLAMA_MODEL"
) &

# This blocks → PID 1 of the container
exec $LMS server start --port 11434 --bind 0.0.0.0
```

Variant B is preferred if it works (cleaner PID 1 handling).

---

## Step 4: scripts/switch-model.sh — hot-swap

Replace the current vLLM restart-based script with hot-swap:

```bash
#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
LMS=/home/user/.lmstudio/bin/lms
MODEL=$1

if [ -z "$MODEL" ]; then
    echo "Verwendung: switch-model.sh <modell-name>"
    echo "Beispiele:"
    echo "  switch-model.sh lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf"
    echo "  switch-model.sh lmstudio-community/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf"
    echo ""
    echo "Aktuell:"
    podman exec llm $LMS ps
    echo ""
    echo "Verfügbar:"
    podman exec llm $LMS ls
    exit 1
fi

# Hot-swap: unload current, load new (no container restart!)
echo "Unloading current model..."
podman exec llm $LMS unload 2>/dev/null || true

echo "Loading: $MODEL"
podman exec llm $LMS load "$MODEL"

# Update .env for next restart
sed -i "s|^LLAMA_MODEL=.*|LLAMA_MODEL=$MODEL|" "$ENV_FILE"

echo "✓ Switched to: $MODEL"
```

---

## Step 5: Update .env and .env.example

`.env`:
```
HF_CACHE=/home/user/.cache/huggingface
LMSTUDIO_MODELS=/home/user/.lmstudio/models

# llama-server model — hot-swap with: scripts/switch-model.sh <name>
LLAMA_MODEL=lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf
```

`.env.example`:
```
HF_CACHE=/home/user/.cache/huggingface
LMSTUDIO_MODELS=/home/user/.lmstudio/models
LLAMA_MODEL=lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf
```

---

## Step 6: Update honcho-fork/.env LLM URLs

Change all `http://llm:8000/v1` → `http://llm:11434/v1` in `honcho-fork/.env`.
(The embeddings URL `http://embeddings:11435/v1` stays unchanged.)

---

## Step 7: Handle API model name

LM Studio's API may return the full model path or an alias as the model `id`.
OpenClaw is configured with `model: "current"`.

**Option A:** lms supports `--alias` or similar → set alias to `current`
Check: `lms server start --help` for alias/name flags.

**Option B:** lms ignores the model name in requests (returns whatever is loaded)
→ no change needed, OpenClaw's `model: "current"` will work.

**Option C:** lms requires exact model name
→ Update OpenClaw config to use the actual model name from `curl /v1/models`.

Verify during Step 1 testing.

---

## Verification

After implementing:

```bash
# 1. Container starts and stays up
podman ps --filter name=llm

# 2. API responds
curl -s http://localhost:11434/v1/models | python3 -m json.tool

# 3. Inference works
time curl -s http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"current","messages":[{"role":"user","content":"Sag hi"}],"stream":false}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])"

# 4. GPU utilization during inference (should be 80-99%)
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader --loop=1

# 5. Embeddings still work
curl -s http://localhost:11435/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"nomic-ai/nomic-embed-text-v1.5","input":"test"}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('dims:', len(d['data'][0]['embedding']))"

# 6. Hot-swap works
~/ai-stack/scripts/switch-model.sh \
  lmstudio-community/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf
```

---

## Open questions (answer in Step 1)

1. Does `lms server start` block or daemonize? → determines entrypoint variant
2. What model `id` does the API return? → determines if OpenClaw config needs update
3. Does lms need write access anywhere besides `.internal`? → refine volume mounts
4. Does `lms load` accept the relative path (from `lms ls`) or full path?
