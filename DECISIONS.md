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

## Context window: 65536 not 150000

OpenClaw uses up to 150k token context. The original Ollama setup defaulted to
150k (`context_length: 150000`) which pre-allocated ~11 GB of KV cache on top of
the ~18 GB model weights — exceeding available VRAM on a single GPU and forcing
everything onto both GPUs with heavy PCIe communication.

`VLLM_MAX_CTX=65536` provides ample context for real conversations (LM Studio
sessions showed ~29k-61k actual usage) while keeping KV cache memory reasonable.
Adjust in `.env` if longer context is needed.

---

## speaches: dynamic GPU selection

`scripts/speaches-entrypoint.sh` detects the GPU with the most free VRAM at
container startup and sets `CUDA_VISIBLE_DEVICES` accordingly. This way
speaches doesn't compete with the LLM for GPU memory on a fixed GPU, but
adapts to whatever is available when the stack starts.

speaches uses ~200 MB VRAM — negligible compared to the LLM.
