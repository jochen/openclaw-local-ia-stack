#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
MODELS_DIR="/home/user/.lmstudio/models"
CMD=$1

print_usage() {
    echo "Verwendung: switch-model.sh <container-pfad>"
    echo "            switch-model.sh search <begriff>"
    echo "Beispiele:"
    echo "  switch-model.sh /models/lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf"
    echo "  switch-model.sh /models/lmstudio-community/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf"
    echo "  switch-model.sh search Q4_K_S"
    echo ""
    CURRENT=$(grep "^LLAMA_MODEL=" "$ENV_FILE" | cut -d= -f2-)
    CURRENT_MMPROJ=$(grep "^LLAMA_MMPROJ=" "$ENV_FILE" | cut -d= -f2-)
    CURRENT_MTP=$(grep "^LLAMA_MTP=" "$ENV_FILE" | cut -d= -f2-)
    echo "Aktuell: $CURRENT"
    echo "  mmproj: $CURRENT_MMPROJ"
    echo "  MTP:    $CURRENT_MTP"
    echo ""
    echo "Verfügbare Modelle:"
    find "$MODELS_DIR" -iname "*.gguf" ! -iname "*mmproj*" 2>/dev/null \
        | sed "s|$MODELS_DIR|/models|" \
        | sort
}

if [ -z "$CMD" ]; then
    print_usage
    exit 1
fi

# ── Subkommando: search <begriff> ────────────────────────────────────────────
# Durchsucht alle .gguf-Dateien (auch noch nicht heruntergeladene werden
# natürlich nicht gefunden) nach <begriff> im Dateinamen, case-insensitive.
# Rein lesend — ändert weder .env noch startet es Container neu.
if [ "$CMD" = "search" ]; then
    TERM=$2
    if [ -z "$TERM" ]; then
        echo "Verwendung: switch-model.sh search <begriff>"
        exit 1
    fi
    find "$MODELS_DIR" -iname "*${TERM}*.gguf" 2>/dev/null \
        | sed "s|$MODELS_DIR|/models|" \
        | sort
    exit 0
fi

MODEL=$CMD

# ── mmproj im selben Verzeichnis suchen ─────────────────────────────────────
HOST_MODEL_PATH="$MODELS_DIR${MODEL#/models}"
MODEL_DIR=$(dirname "$HOST_MODEL_PATH")

MMPROJ_MATCHES=()
if [ -d "$MODEL_DIR" ]; then
    while IFS= read -r -d '' f; do
        MMPROJ_MATCHES+=("$f")
    done < <(find "$MODEL_DIR" -maxdepth 1 -iname "*mmproj*.gguf" -print0 2>/dev/null)
fi

if [ "${#MMPROJ_MATCHES[@]}" -eq 1 ]; then
    MMPROJ_CONTAINER="/models${MMPROJ_MATCHES[0]#$MODELS_DIR}"
    sed -i "s|^LLAMA_MMPROJ=.*|LLAMA_MMPROJ=$MMPROJ_CONTAINER|" "$ENV_FILE"
    echo "mmproj gefunden, gesetzt auf: $MMPROJ_CONTAINER"
elif [ "${#MMPROJ_MATCHES[@]}" -eq 0 ]; then
    sed -i "s|^LLAMA_MMPROJ=.*|LLAMA_MMPROJ=|" "$ENV_FILE"
    echo "Kein mmproj im Modellverzeichnis gefunden — LLAMA_MMPROJ geleert."
else
    echo "Mehrere mmproj-Kandidaten gefunden — bitte LLAMA_MMPROJ manuell in .env setzen:"
    for f in "${MMPROJ_MATCHES[@]}"; do
        echo "  /models${f#$MODELS_DIR}"
    done
fi

# ── MTP-Unterstützung heuristisch erkennen ──────────────────────────────────
if echo "$MODEL" | grep -qi "mtp"; then
    sed -i "s|^LLAMA_MTP=.*|LLAMA_MTP=true|" "$ENV_FILE"
    echo "Modellname enthält 'MTP' — LLAMA_MTP=true gesetzt."
else
    sed -i "s|^LLAMA_MTP=.*|LLAMA_MTP=false|" "$ENV_FILE"
fi

sed -i "s|^LLAMA_MODEL=.*|LLAMA_MODEL=$MODEL|" "$ENV_FILE"

echo "Switching to: $MODEL"
podman compose -f "$SCRIPT_DIR/../compose.yml" up -d --force-recreate llm
echo "✓ llm-Container neu gestartet"
