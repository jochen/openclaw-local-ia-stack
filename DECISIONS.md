# Architecture Decisions & Lessons Learned

This document captures why the stack is built the way it is — specifically the dead ends
we explored and what we learned from them.

---

## Why not LM Studio?

LM Studio was the original LLM backend. It worked well performance-wise, but:

- **GUI-only** — no headless/SSH operation without `DISPLAY` environment
- **No CLI model management** — model switching requires GUI interaction
- **Hard to script** — no way to trigger model loads via API or CLI remotely

LM Studio remains the **fallback** if everything else breaks, since it proved
the hardware can run the models at acceptable speed.

---

## Why not Ollama?

Ollama was the first attempted replacement. It has excellent SSH/CLI support,
built-in model management, and OpenAI-compatible API. We discovered three blockers:

### 1. Multi-GPU: Tensor Parallelism over PCIe is too slow

The RTX 5060 Ti (Blackwell) has no NVLink. Ollama uses **tensor parallelism**
by default — every token generation requires ALL-REDUCE operations between GPUs
over PCIe. This resulted in **~5-9% GPU utilization** during inference and
**4-6 minute response times** for a simple joke.

Evidence from `nvidia-smi` GPU utilization log:
```
21:34:21–21:35:31  GPU0: 5-9%  GPU1: 5-9%   → token generation (70 seconds)
21:35:32–21:36:05  GPU0: 94%   GPU1: 70-99% → prefill burst
21:36:06–21:39:15  GPU0: 6-7%  GPU1: 6-7%   → token generation (3.5 minutes!)
```

The model LM Studio was using (same hardware, same model) ran fast because
LM Studio uses **pipeline parallelism** — GPU0 handles layers 0-24, GPU1 handles
layers 25-48. Only the activations are transferred between GPUs once per layer,
not a full ALL-REDUCE per layer. Dramatically less PCIe traffic per token.

**Key learning:** On PCIe-only multi-GPU setups (no NVLink), pipeline parallelism
is essential. Ollama does not support this.

### 2. The model we wanted doesn't exist in Ollama

`qwen3.5-35b-a3b` was the model running in LM Studio. The closest in Ollama
was `qwen3:30b-a3b` — a different model. The exact model is available as AWQ
on HuggingFace but not in Ollama's registry.

### 3. Known vision issues

Ollama has an open GitHub issue affecting vision model inference. Since vision
is a required feature (image analysis in OpenClaw), this was a hard blocker.

---

## Why vLLM?

vLLM supports `--pipeline-parallel-size 2`, which distributes transformer layers
across GPUs in pipeline fashion. This matches how LM Studio achieved good performance
on the same hardware.

Additional advantages:
- `--served-model-name current,vision` — both API aliases work with one service
- AWQ quantization supported natively
- `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit` is the exact model family previously used
- Official Docker image with CUDA 12.9 support (`cu129-nightly`) for Blackwell GPUs

### vLLM image tag note

Only `cu129-nightly` and `nightly` tags exist (no `latest` stable). This is because
the RTX 5060 Ti (Blackwell) requires CUDA 12.9, which was still in nightly at the
time this was set up. The nightly tag is rebuilt daily and is stable enough for
production use.

---

## Why AWQ over GGUF?

Both GGUF (used by Ollama/llama-server) and AWQ are ~4-bit quantizations.
For vLLM, AWQ is the native format — no conversion needed since the models
exist on HuggingFace in AWQ format already.

GGUF → AWQ conversion is not straightforward (requires dequantization and
re-quantization with calibration data). The pre-existing AWQ models from
`cyankiwi` on HuggingFace have 479k downloads and are the de-facto community
standard for this model family.

---

## Why infinity-emb instead of text-embeddings-inference (TEI)?

Both were evaluated for serving `nomic-ai/nomic-embed-text-v1.5` (768-dim
embedding model).

TEI failed with a parse error on both `v1` and `v1.5`:
```
Error: Failed to parse `config.json`
Caused by: duplicate field `max_position_embeddings`
```

The nomic-embed-text models have duplicate fields in their `config.json` that
TEI's strict parser rejects. `infinity-emb` handles this gracefully.

---

## Why 768 dimensions instead of the default 1536?

Honcho defaults to OpenAI's `text-embedding-3-small` (1536 dims). We use
`nomic-embed-text-v1.5` locally (768 dims) to avoid cloud API calls.

This required:
1. Patching `src/models.py` in honcho-fork: `Vector(1536)` → `Vector(_VECTOR_DIMS)`
   where `_VECTOR_DIMS = settings.EMBEDDING.VECTOR_DIMENSIONS`
2. Running `ALTER TABLE` to change existing DB columns:
   ```sql
   ALTER TABLE message_embeddings ALTER COLUMN embedding TYPE vector(768);
   ALTER TABLE documents ALTER COLUMN embedding TYPE vector(768);
   ```
3. Removing an overly strict validator in `config.py` that prevented
   non-1536 dimensions when using pgvector (the guard assumed 1536 = always correct,
   but we intentionally set up the DB with 768 dims from the start)

These patches live in [jochen/honcho](https://github.com/jochen/honcho) on `main`.

---

## Why vLLM failed: AWQ model too large for pre-allocated KV cache

vLLM was attempted as the replacement for Ollama. It supports both
`--pipeline-parallel-size 2` and `--tensor-parallel-size 2` with NCCL communication.
Both modes failed with CUDA out of memory errors, for the same root cause.

### The fundamental mismatch

The AWQ compressed-tensors version of `Qwen3.5-35B-A3B` (`cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit`)
is **22.78 GiB** on disk (vs 18 GB for the GGUF Q4_K_M). Loaded into VRAM with TP=2,
each GPU ends up with ~14.4 GiB of model weights — leaving only ~1 GiB free per GPU.

vLLM uses **eager KV cache pre-allocation**: it reserves a pool of KV cache blocks
upfront based on `gpu-memory-utilization` × available_memory. With only 1 GiB free,
this pool is far too small for any meaningful context window.

Attempted configurations and their failures:

**Pipeline parallel (`--pipeline-parallel-size 2`):**
```
PP stage 0 (GPU 0): 14.8 GiB allocated → only 88 MiB free
Tried to allocate 1.03 GiB for KV cache → OOM
```
PP stage 0 received the Vision Encoder + its layer range, which is disproportionately
large for a multimodal MoE model.

**Tensor parallel (`--tensor-parallel-size 2`):**
```
Each GPU: ~14.4 GiB model weights → only ~1 GiB free
Tried to allocate 528 MiB for KV cache profiling → OOM
```
Even with equal weight distribution, the Vision Encoder still loaded predominantly
on GPU 0.

### Why GGUF + llama.cpp (LM Studio) worked

LM Studio ran the GGUF version (18 GB on disk, ~26 GB in VRAM) with 150K context
because llama.cpp uses **lazy KV cache allocation**: it allocates KV cache blocks
on demand per active sequence rather than pre-reserving a large pool.

Observed VRAM in LM Studio:
- GPU 0: 14.6 GiB
- GPU 1: 11.7 GiB
- KV cache: allocated dynamically on top, fits because it grows gradually

This is a fundamental architectural difference between llama.cpp (single-user,
lazy allocation) and vLLM (throughput-optimized, eager pool allocation). For a
single-user setup like OpenClaw + honcho, llama.cpp's model is strictly superior.

### vLLM compilation overhead (bonus learning)

vLLM with CUDA graph compilation (`CompilationMode.VLLM_COMPILE`) takes 20-40 minutes
on first startup to compile Torch Inductor kernels for 51 batch sizes. The cache
must be persisted in a named volume (`vllm-cache`) or recompilation happens on
every container restart. Even with the cache, startup time is significant.

---

## Why lms (LM Studio CLI) was not used — and what we use instead

After vLLM failed, the plan was to containerize LM Studio's own inference engine via
the `lms` CLI. Testing revealed a hard blocker:

```
$ lms server start --port 11434 --bind 0.0.0.0
Error: LM Studio daemon is not running and no valid installation could be found or installed.
```

`lms server start` is **not a standalone server**. It's a control-plane CLI that
requires the LM Studio desktop app daemon to be running. Without the GUI app, `lms`
has nothing to connect to. The bundled backends in
`~/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.13.0/`
are `.so` shared libraries — no standalone executable.

### What we use: `ghcr.io/ggml-org/llama.cpp:server-cuda`

The official llama.cpp server image directly serves the OpenAI-compatible API.
Key flags:

```
--split-mode=layer   # pipeline parallelism: GPU0 handles layers 0..N/2, GPU1 N/2..end
--n-gpu-layers=99    # offload all layers to VRAM
--alias=current      # API clients always send model="current" regardless of GGUF file
```

`--split-mode=layer` is the pipeline-parallelism mode that LM Studio used internally,
which is why LM Studio performed well on this hardware (see "Why not Ollama?" above).

**Hot-swap:** Without `lms load/unload`, model switching requires a container restart
(`scripts/switch-model.sh`). For single-user use this is acceptable — restart takes
~30 seconds, not 20 minutes like vLLM's Torch compilation.

---

## Context window and KV cache

LM Studio sessions showed actual usage of 29k–61k tokens. The model supports
up to 128K context via YaRN. The llama-server flag `--ctx-size` sets the maximum
context window; actual KV cache only grows to the active context length.

`LLAMA_CTX=131072` in `.env` allows up to 131K context (matching LM Studio's
observed 150K configuration) without pre-allocating all of it.

---

## speaches: fixed GPU assignment (was: dynamic selection)

**Superseded 2026-07-05.** The original approach — `speaches-entrypoint.sh`
picks the GPU with the most free VRAM at container startup — broke once the
LLM presets moved to ctx 131072: the choice is made *once at start*, but the
LLM loads lazily *afterwards* and fills whichever GPU speaches picked.
Result: `CUDA out of memory` when Whisper loaded on first transcription →
HTTP 500 in the voice assistant. The "~200 MB VRAM" assumption was also
wrong: faster-whisper-medium int8 needs ~1.2 GB resident (with
`WHISPER__TTL=-1`), plus ~1.5 GB peak during ctranslate2 model load.

**Current design:** GPU1 is the *service GPU* with a fixed budget:
SER (wav2vec2-large fp16, ~1.6 GB) + Whisper (~1.2 GB) + speaches/piper
(~0.5 GB) ≈ 3.3 GB. speaches is pinned via `SPEACHES_GPU=1` (compose.yml →
entrypoint honors it; dynamic selection remains as fallback if unset).
The LLM is shifted toward GPU0 via `tensor-split = 59,41` in
`llm-presets.ini` `[*]`, capping its GPU1 share at ~12 GB. Verified with
gemma/qwen/ornith at ctx 131072: STT/TTS/SER/voice-analysis all pass while
the LLM is loaded. bge-m3 (infinity) turned out to run on CPU anyway and
needs no GPU budget.

Note: gemma's preset is text-only now (mmproj costs 1.1 GB it no longer
has); vision requests use the `[gemma-vision]` preset (ctx 65536 + MTP).

---

## Qwen3 thinking mode breaks honcho deriver and summarizer

Qwen3.5-35B-A3B is a reasoning model. It generates `<think>...</think>` tokens
before every response. These thinking tokens count toward the `max_tokens` budget
in the llama.cpp API.

Honcho's deriver and summarizer use small token budgets (`DEFAULT_MAX_TOKENS=2500`,
`MAX_TOKENS_SHORT=1000`) sized for fast, focused output — not for reasoning chains.
The model was spending the entire budget on thinking and returning empty content,
causing:

```
ERROR - ❌ Repair failed: Expecting value: line 1 column 1 (char 0)
ERROR - Generated summary is empty (finish_reasons=['length'])
```

### What doesn't work

**`reasoning_effort: "none"`** (standard OpenAI API parameter): llama.cpp silently
ignores this for Qwen3. The model continues to think. Verified by direct API test:
`content` was empty, `reasoning_content` contained the full thinking trace.

### What works

**`chat_template_kwargs: {"enable_thinking": false}`** in the request body: llama.cpp
maps this to the Qwen3 `<|nothink|>` token in the chat template, fully suppressing
thinking before generation starts. Verified: content returned immediately, no
`reasoning_content`.

### Implementation

The OpenAI client SDK rejects unknown top-level parameters. `chat_template_kwargs`
must be passed via `extra_body` (which the SDK forwards to the JSON body without
validation):

```python
await client.chat.completions.create(**params, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
```

This is wired in `honcho-fork/src/llm/backends/openai.py`: when `thinking_effort == "none"`,
`_build_params` sets a `_extra_body` sentinel instead of `reasoning_effort`, which the
caller pops and passes as `extra_body`.

Activated in `honcho-fork/.env` via:
```
DERIVER_MODEL_CONFIG__THINKING_EFFORT=none
SUMMARY_MODEL_CONFIG__THINKING_EFFORT=none
```

### Model compatibility

| Model type | Effect of `enable_thinking: false` |
|------------|-------------------------------------|
| Qwen3 family (thinking models) | Thinking suppressed — correct behavior |
| Qwen2.5, Llama 3, Mistral, etc. | Silently ignored by llama.cpp — no effect |
| DeepSeek-R1 via llama.cpp | Also supported via same mechanism |
| Real OpenAI o1/o3 API | Would break — but we use local llama.cpp only |

If the active model is switched to a non-thinking model, the setting is harmless
and can be left as-is.

---

## speaches: Blackwell PTX JIT and CUDA cache

The speaches image (`latest-cuda`) was built with CUDA 12.6 and does not include
compiled kernels for sm_120 (Blackwell). On first use of faster-whisper, CUDA
JIT-compiles the kernels at runtime, which takes ~50 seconds.

**Fix:** `CUDA_CACHE_PATH` is set to a named Podman volume (`speaches-cuda-cache`).
CUDA stores compiled PTX kernels there, so subsequent container restarts skip
recompilation. First-ever start is slow; all subsequent starts are fast (<0.5s).

**STT model loading** is handled by the `speaches-warmup` sidecar container, which
POSTs to `/v1/models/{model_id}` after speaches is healthy. This triggers model
download (first run) or cache load (subsequent runs), ensuring OpenClaw finds the
models in `/v1/models` immediately without falling back.

The model list (`STT_MODEL`, `TTS_MODEL`) lives in `compose.yml` under
`speaches-warmup.environment` — change it there to use different models.

---

## rouven: Speech-Stack ausgelagert (speaches + ser + voice-analysis)

Um GPU1 auf diesem Host (126) komplett dem LLM zu überlassen (VRAM-Konflikte,
siehe Speaches-TTL-Eintrag oben), laufen `speaches`, `ser` und `voice-analysis`
testweise auf einem zweiten Fablab-Rechner ("rouven", 192.168.111.229, GTX 1660
6GB). Config: `compose.rouven.yml` (gleiche `./ser` und `./voice-analysis`
Build-Contexts wie hier, deshalb im selben Verzeichnis statt in einem
Unterordner). Umschaltbar über `switch-audio-stack.sh` im
`openclaw_voice_assist`-Repo.

**Setup-Voraussetzungen auf einem neuen Host (Ubuntu 24.04, rootless Podman):**
- `nvidia-container-toolkit` MUSS auf 1.16.x gepinnt werden (`apt-get install
  --allow-downgrades nvidia-container-toolkit=1.16.2-1 ...`). Ab 1.17+ generiert
  `nvidia-ctk cdi generate` ein `additionalGids`-Feld, das podman 4.9.3 (Ubuntu
  24.04 Standard) nicht parsen kann → "unresolvable CDI devices". Downgrade
  erzeugt kompatibles CDI ohne das Feld.
- `loginctl enable-linger <user>` ist Pflicht — ohne Linger sterben
  rootless-Podman-Container beim Ende der SSH-Session (hängen dann in
  "Stopping" fest, `podman system migrate` + `podman rm -f` + `podman network
  prune -f` nötig zum Aufräumen).

**GPU-Kompatibilität (kritisch!):** Die GTX 1660 hat keine Tensor Cores (im
Gegensatz zur RTX 5060 Ti hier). `WHISPER__COMPUTE_TYPE=int8` oder `float16`
liefern auf dieser Karte STILLSCHWEIGEND falschen Text — keine Fehlermeldung,
kein Crash, nur plausibel klingende Halluzinationen (int8) bzw. Garbage-Tokens
(float16). Nur mit CPU-Fallback-Vergleich (identische Audiodatei, einmal
`WHISPER__INFERENCE_DEVICE=cpu`) auffindbar. Fix: `WHISPER__COMPUTE_TYPE=float32`
— langsamer (RTF ~0.78 statt ~0.13), aber korrekt. VRAM-Verbrauch mit `ser`
zusammen: ~4.2/6GB, noch Puffer.

**Diarization-Modelle** (`fedirz/segmentation_community_1` +
`Wespeaker/wespeaker-voxceleb-resnet34-LM`) müssen wie STT/TTS explizit über
`speaches-warmup` registriert werden, sonst 404 "not installed locally" beim
ersten `/v1/audio/diarization`-Aufruf. In `compose.rouven.yml` ergänzt.
