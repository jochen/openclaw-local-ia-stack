#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
MODEL=$1

if [ -z "$MODEL" ]; then
    echo "Verwendung: switch-model.sh <modell-pfad-im-container>"
    echo "Beispiele:"
    echo "  switch-model.sh /models/lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf"
    echo "  switch-model.sh /models/lmstudio-community/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf"
    echo ""
    CURRENT=$(grep "^LLAMA_MODEL=" "$ENV_FILE" | cut -d= -f2-)
    echo "Aktuell: $CURRENT"
    echo ""
    echo "Verfügbare Modelle:"
    find /home/user/.lmstudio/models -name "*.gguf" 2>/dev/null | sed 's|/home/user/.lmstudio/models|/models|'
    exit 1
fi

sed -i "s|^LLAMA_MODEL=.*|LLAMA_MODEL=$MODEL|" "$ENV_FILE"

echo "Switching to: $MODEL"
~/.local/bin/podman-compose -f "$SCRIPT_DIR/../compose.yml" up -d --force-recreate llm
echo "✓ llama-server neu gestartet"
