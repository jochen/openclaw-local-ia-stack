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

## Why lms (LM Studio CLI) in Podman

After vLLM failed, the conclusion is to containerize LM Studio's own inference engine.
LM Studio 0.4.x includes an `lms` CLI binary that starts a headless server — solving
the original problems with LM Studio (GUI, no SSH):

```
lms server start --port 11434 --bind 0.0.0.0  # headless, no display needed
lms load <model>                                # hot-swap model without restart
lms unload                                      # unload current model
```

**Why this is better than building a new solution:**
- The inference engine (llama.cpp v2.13.0 with CUDA 12) is already proven to work
  on this exact hardware with this exact model
- LM Studio bundles its own CUDA libraries (`libcublas.so.12`, `libcublasLt.so.12`,
  ~870 MB) — the container only needs NVIDIA driver access, not the host CUDA Toolkit
- No new software to install, no compilation, no unknown variables
- Hot-swap model switching via `lms load/unload` — works without container restart

The `lms` binary is a self-contained native ELF with only standard glibc dependencies.
Containerization: mount the LM Studio installation at the same path, use a named
volume for runtime state (`~/.lmstudio/.internal`).

---

## Context window and KV cache

LM Studio sessions showed actual usage of 29k–61k tokens. The model supports
up to 128K context via YaRN. The llama-server flag `--ctx-size` sets the maximum
context window; actual KV cache only grows to the active context length.

`LLAMA_CTX=131072` in `.env` allows up to 131K context (matching LM Studio's
observed 150K configuration) without pre-allocating all of it.

---

## speaches: dynamic GPU selection

`scripts/speaches-entrypoint.sh` detects the GPU with the most free VRAM at
container startup and sets `CUDA_VISIBLE_DEVICES` accordingly. This way
speaches doesn't compete with the LLM for GPU memory on a fixed GPU, but
adapts to whatever is available when the stack starts.

speaches uses ~200 MB VRAM — negligible compared to the LLM.
