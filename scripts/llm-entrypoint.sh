#!/bin/bash
set -e

# Baut die llama-server Argumentliste dynamisch aus Umgebungsvariablen auf.
# Erlaubt das Wechseln von Modell/mmproj/Kontextgröße/MTP-Spekulation
# ohne compose.yml anpassen zu müssen (siehe scripts/switch-model.sh).

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
  --parallel 2
  -ngl 99
  -t 6
  -tb 6
  --flash-attn on
)

# Multi-Token-Prediction (MTP) Spekulationsdecoding: nur aktivieren,
# wenn LLAMA_MTP=true/1 gesetzt ist. Nutzt die im Hauptmodell
# vorhandenen MTP-Tensoren selbst-spekulativ (kein separates Draft-Modell).
if [ "${LLAMA_MTP}" = "true" ] || [ "${LLAMA_MTP}" = "1" ]; then
  ARGS+=( --spec-type draft-mtp )
fi

echo "llm-entrypoint: starte llama-server mit:"
printf '  %s\n' "${ARGS[@]}"

exec /app/llama-server "${ARGS[@]}"
