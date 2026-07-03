#!/bin/bash
set -e

# Baut die llama-server Argumentliste dynamisch aus Umgebungsvariablen auf.
# Erlaubt das Wechseln von Modell/mmproj/Kontextgröße/MTP-Spekulation
# ohne compose.yml anpassen zu müssen (siehe scripts/switch-model.sh).

# Router-Mode: wenn LLAMA_PRESETS gesetzt ist, verwaltet llama-server mehrere
# Modelle aus der INI-Datei und lädt/entlädt sie bei Bedarf (LRU, --models-max).
# Modellname im Request = INI-Sektionsname. Alle Modell-Parameter stehen
# dann in der Preset-Datei, die LLAMA_MODEL/LLAMA_CTX/...-Vars gelten NICHT.
if [ -n "${LLAMA_PRESETS}" ]; then
  ARGS=(
    --models-preset "${LLAMA_PRESETS}"
    --models-max "${LLAMA_MODELS_MAX:-1}"
    --host 0.0.0.0
    --port 11434
  )
  echo "llm-entrypoint: starte llama-server im ROUTER-MODE mit:"
  printf '  %s\n' "${ARGS[@]}"
  exec /app/llama-server "${ARGS[@]}"
fi

ARGS=(
  --model "${LLAMA_MODEL}"
  --alias current
)

# mmproj ist optional: nur setzen, wenn LLAMA_MMPROJ nicht leer ist
if [ -n "${LLAMA_MMPROJ}" ]; then
  ARGS+=( --mmproj "${LLAMA_MMPROJ}" )
fi

ARGS+=(
  --n-gpu-layers 99
  --split-mode layer
  --ctx-size "${LLAMA_CTX}"
  --host 0.0.0.0
  --port 11434
  --parallel "${LLAMA_PARALLEL:-1}"
  -ngl 99
  -t 6
  -tb 6
  --flash-attn on
)

# KV-Cache-Quantisierung: spart ~50% KV-VRAM (q8_0 statt f16) und erlaubt
# damit größere Kontexte. Braucht Flash-Attention (oben aktiv).
# Achtung: mit quantisiertem KV-Cache funktioniert Context-Shift nicht.
if [ -n "${LLAMA_CACHE_K}" ]; then
  ARGS+=( --cache-type-k "${LLAMA_CACHE_K}" )
fi
if [ -n "${LLAMA_CACHE_V}" ]; then
  ARGS+=( --cache-type-v "${LLAMA_CACHE_V}" )
fi

# Multi-Token-Prediction (MTP) Spekulationsdecoding: nur aktivieren,
# wenn LLAMA_MTP=true/1 gesetzt ist. Nutzt die im Hauptmodell
# vorhandenen MTP-Tensoren selbst-spekulativ (kein separates Draft-Modell).
if [ "${LLAMA_MTP}" = "true" ] || [ "${LLAMA_MTP}" = "1" ]; then
  ARGS+=( --spec-type draft-mtp )
fi

echo "llm-entrypoint: starte llama-server mit:"
printf '  %s\n' "${ARGS[@]}"

exec /app/llama-server "${ARGS[@]}"
