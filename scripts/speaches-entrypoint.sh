#!/bin/sh
set -e

# Feste GPU-Zuweisung via SPEACHES_GPU (compose.yml) hat Vorrang.
# Die dynamische Wahl "meiste freie VRAM beim Start" war fehleranfaellig:
# direkt nach Stack-Start gewinnt immer die GPU, die spaeter das LLM
# vollpackt -> Whisper-OOM/500er, sobald ctx 131072 geladen ist.
if [ -n "$SPEACHES_GPU" ]; then
  export CUDA_VISIBLE_DEVICES=$SPEACHES_GPU
  echo "speaches: GPU $SPEACHES_GPU fest zugewiesen (SPEACHES_GPU)"
else
  BEST_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | awk 'NR==1{max=$1;g=0} $1>max{max=$1;g=NR-1} END{print g}')
  export CUDA_VISIBLE_DEVICES=$BEST_GPU
  echo "speaches: GPU $BEST_GPU gewählt (meiste freie VRAM)"
fi

exec /opt/nvidia/nvidia_entrypoint.sh "$@"
