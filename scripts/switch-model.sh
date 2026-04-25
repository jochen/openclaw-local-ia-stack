#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
MODEL=$1

if [ -z "$MODEL" ]; then
    echo "Verwendung: switch-model.sh <hf-model-id>"
    echo "Beispiele:"
    echo "  switch-model.sh cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit"
    echo "  switch-model.sh google/gemma-4-27b-it"
    echo ""
    CURRENT=$(grep "^VLLM_MODEL=" "$ENV_FILE" | cut -d= -f2-)
    echo "Aktuell: $CURRENT"
    exit 1
fi

# .env aktualisieren
sed -i "s|^VLLM_MODEL=.*|VLLM_MODEL=$MODEL|" "$ENV_FILE"

echo "Switching to: $MODEL (quantization: auto-detected by vLLM)"
~/.local/bin/podman-compose -f "$SCRIPT_DIR/../compose.yml" up -d --force-recreate llm
echo "✓ vLLM neu gestartet — erstes Request triggert Model-Load aus HF Cache"
