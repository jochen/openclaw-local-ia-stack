#!/bin/bash
# Schaltet die Sprach-Pipeline (speaches, ser, voice-analysis) ab/an, damit
# der llm-Container das GPU1-VRAM, das speaches+ser sonst belegen, mitnutzen
# kann. voice-analysis haengt von beiden ab und ist ohne sie ohnehin nutzlos.
# embeddings bleibt bewusst aussen vor (wird separat genutzt).
set -euo pipefail

CMD="${1:-}"

case "$CMD" in
  stop)
    echo "Stoppe Sprach-Pipeline (voice-analysis, ser, speaches)..."
    podman stop voice-analysis ser speaches
    echo "✓ gestoppt. VRAM auf GPU1 ist jetzt frei fuer llm."
    ;;
  start)
    echo "Starte Sprach-Pipeline (speaches, ser, voice-analysis)..."
    podman start speaches ser voice-analysis
    echo "✓ gestartet. Health-Status pruefen mit: $0 status"
    ;;
  status)
    podman ps -a --filter name=speaches --filter name=ser --filter name=voice-analysis --filter name=llm       --format 'table {{.Names}}\t{{.Status}}'
    echo ""
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
    ;;
  *)
    echo "Verwendung: $0 {stop|start|status}" >&2
    echo "  stop   = speaches+ser+voice-analysis stoppen (VRAM frei fuer llm)" >&2
    echo "  start  = alle drei wieder hochfahren" >&2
    echo "  status = Container-Status + GPU-VRAM anzeigen" >&2
    exit 1
    ;;
esac
