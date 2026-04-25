#!/bin/sh
set -e

BEST_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
  | awk 'NR==1{max=$1;g=0} $1>max{max=$1;g=NR-1} END{print g}')
export CUDA_VISIBLE_DEVICES=$BEST_GPU
echo "speaches: GPU $BEST_GPU gewählt (meiste freie VRAM)"

exec /opt/nvidia/nvidia_entrypoint.sh "$@"
