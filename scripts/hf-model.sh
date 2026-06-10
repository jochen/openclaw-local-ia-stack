#!/bin/bash
set -e

MODELS_DIR="/home/user/.lmstudio/models"

print_usage() {
    echo "Verwendung:"
    echo "  hf-model.sh search <repo> [filter]"
    echo "      Listet alle .gguf-Dateien eines HF-Repos mit Größe."
    echo "      Optionaler [filter] schränkt per Substring (z.B. Q4_K_S) ein."
    echo ""
    echo "  hf-model.sh download <repo> <dateiname>"
    echo "      Lädt eine einzelne Datei nach"
    echo "      $MODELS_DIR/<repo>/<dateiname>"
    echo "      (gleiche Verzeichnisstruktur wie LM Studio, passt zu switch-model.sh)"
    echo ""
    echo "Beispiele:"
    echo "  hf-model.sh search llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF"
    echo "  hf-model.sh search llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF Q4_K"
    echo "  hf-model.sh download llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q4_K_S.gguf"
}

human_size() {
    # Bytes -> GiB mit 1 Nachkommastelle
    awk -v b="$1" 'BEGIN { printf "%.1f GiB", b/1024/1024/1024 }'
}

CMD=$1
REPO=$2

if [ -z "$CMD" ] || [ -z "$REPO" ]; then
    print_usage
    exit 1
fi

case "$CMD" in
  search)
    FILTER=$3
    DATA=$(curl -sf "https://huggingface.co/api/models/${REPO}?blobs=true")
    if [ -z "$DATA" ] || [ "$(echo "$DATA" | jq -r 'has("siblings")')" != "true" ]; then
        echo "Fehler: Repo '$REPO' nicht gefunden oder keine Antwort von huggingface.co" >&2
        exit 1
    fi
    echo "Verfügbare .gguf-Dateien in $REPO:"
    echo "$DATA" | jq -r '.siblings[] | select(.rfilename | endswith(".gguf")) | "\(.size)\t\(.rfilename)"' \
        | sort -k2 \
        | while IFS=$'\t' read -r size name; do
            if [ -n "$FILTER" ] && [[ "$name" != *"$FILTER"* ]]; then
                continue
            fi
            printf "  %-9s  %s\n" "$(human_size "$size")" "$name"
          done
    ;;

  download)
    FILE=$3
    if [ -z "$FILE" ]; then
        echo "Verwendung: hf-model.sh download <repo> <dateiname>"
        exit 1
    fi
    DEST="$MODELS_DIR/$REPO"
    mkdir -p "$DEST"
    echo "Lade $REPO/$FILE nach $DEST/ ..."
    hf download "$REPO" "$FILE" --local-dir "$DEST"
    echo "✓ fertig: $DEST/$FILE"
    echo ""
    echo "Container-Pfad für switch-model.sh:"
    echo "  /models/$REPO/$FILE"
    ;;

  *)
    print_usage
    exit 1
    ;;
esac
