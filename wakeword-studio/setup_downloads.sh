#!/bin/bash
# Trainings-Umgebung + Datensätze für openwakeword-Training (Gaston, M2).
# Läuft als nohup-Job auf dem ai-stack; Log: setup_downloads.log
set -o pipefail
cd ~/wakeword-studio
PIP=./venv/bin/pip
PY=./venv/bin/python
FAIL=0

stage() { echo; echo "===== [$(date +%H:%M:%S)] $1 ====="; }

stage "1/6 pip: openwakeword (editable) + Trainings-Deps"
$PIP install -q -e ./openwakeword || FAIL=1
$PIP install -q mutagen torchinfo torchmetrics speechbrain audiomentations \
  torch-audiomentations acoustics pronouncing datasets scipy tqdm soundfile || FAIL=1

stage "2/6 openwakeword Basis-Modelle (embedding/melspectrogram)"
mkdir -p openwakeword/openwakeword/resources/models
for f in embedding_model.onnx embedding_model.tflite melspectrogram.onnx melspectrogram.tflite; do
  [ -s "openwakeword/openwakeword/resources/models/$f" ] || \
    wget -q -O "openwakeword/openwakeword/resources/models/$f" \
    "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/$f" || FAIL=1
done

stage "3/6 Vorgerechnete Features (ACAV100M ~16GB + Validation)"
[ -s openwakeword_features_ACAV100M_2000_hrs_16bit.npy ] || \
  wget -q -O openwakeword_features_ACAV100M_2000_hrs_16bit.npy \
  "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy" || FAIL=1
[ -s validation_set_features.npy ] || \
  wget -q -O validation_set_features.npy \
  "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy" || FAIL=1

stage "4/6 MIT Room Impulse Responses"
$PY download_bg.py rirs || FAIL=1

stage "5/6 AudioSet-Teil (bal_train09) + 16k-Konvertierung"
if [ ! -d audioset_16k ] || [ -z "$(ls -A audioset_16k 2>/dev/null)" ]; then
  mkdir -p audioset
  [ -s audioset/bal_train09.tar ] || wget -q -O audioset/bal_train09.tar \
    "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train09.tar" || FAIL=1
  (cd audioset && tar -xf bal_train09.tar) || FAIL=1
  $PY download_bg.py audioset || FAIL=1
fi

stage "6/6 FMA-Musik-Subset (2h, 16k)"
$PY download_bg.py fma || FAIL=1

stage "Ergebnis"
ls -sh openwakeword_features_ACAV100M_2000_hrs_16bit.npy validation_set_features.npy 2>/dev/null
for d in mit_rirs audioset_16k fma; do
  echo "$d: $(ls "$d" 2>/dev/null | wc -l) Dateien"
done
echo "FAIL=$FAIL"
exit $FAIL
