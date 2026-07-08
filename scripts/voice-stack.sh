#!/bin/bash
# Schaltet die Sprach-Pipeline (speaches, ser, voice-analysis) UND das
# llm-Context-Preset gemeinsam um:
#   stop  = Sprach-Pipeline aus + llm auf volle ctx 131072 + mmproj
#           (llm-presets.ini.full, kein tensor-split-Reserve)
#   start = llm auf kompakte ctx 65536 mit GPU1-Reserve
#           (llm-presets.ini.compact, tensor-split=59,41) + Sprach-Pipeline an
# embeddings bleibt bewusst aussen vor (wird separat genutzt).
#
# llama-server liest llm-presets.ini nur beim Start ein (Bind-Mount, kein
# Live-Reload) -> switch_llm kopiert IMMER + recreated IMMER, keine
# cmp-Abkuerzung (fuehrte schon mal dazu, dass die Datei zwar aktuell war,
# der laufende Container aber noch die alte Config im Speicher hatte).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STACK_DIR="$(dirname "$SCRIPT_DIR")"
PRESETS_LIVE="$STACK_DIR/llm-presets.ini"

CMD="${1:-}"

switch_llm() {
  # $1 = full|compact
  local src="$STACK_DIR/llm-presets.ini.$1"
  cp "$src" "$PRESETS_LIVE"
  echo "llm-presets.ini -> $1, recreate llm..."
  "$HOME/.local/bin/podman-compose" -f "$STACK_DIR/compose.yml" up -d --force-recreate llm >/dev/null
  echo "✓ llm neu gestartet mit '$1'-Preset"
}

case "$CMD" in
  stop)
    echo "Stoppe Sprach-Pipeline (voice-analysis, ser, speaches)..."
    podman stop voice-analysis ser speaches
    switch_llm full
    echo "✓ VRAM frei fuer llm (ctx 131072 + mmproj)."
    ;;
  start)
    switch_llm compact
    echo "Starte Sprach-Pipeline (speaches, ser, voice-analysis)..."
    podman start speaches ser voice-analysis
    echo "✓ gestartet. Health-Status pruefen mit: $0 status"
    ;;
  status)
    podman ps -a --filter name=speaches --filter name=ser --filter name=voice-analysis --filter name=llm       --format 'table {{.Names}}\t{{.Status}}'
    echo ""
    echo "llm ctx-size (aktives Preset, [qwen]):"
    awk '/^\[qwen\]/{f=1} f && /^ctx-size/{print "  " $0; exit}' "$PRESETS_LIVE"
    echo ""
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
    ;;
  *)
    echo "Verwendung: $0 {stop|start|status}" >&2
    echo "  stop   = Sprach-Pipeline stoppen + llm auf volle ctx 131072/mmproj" >&2
    echo "  start  = llm auf kompakte ctx 65536 (GPU1-Reserve) + Sprach-Pipeline hochfahren" >&2
    echo "  status = Container-Status + aktives ctx-Preset + GPU-VRAM anzeigen" >&2
    exit 1
    ;;
esac
