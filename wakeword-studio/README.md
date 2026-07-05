# Wakeword-Studio — Trainings-Pipeline (openWakeWord, deutsche Stimmen)

Trainiert eigene Wakeword-Modelle für den OpenClaw-Voice-Assistant
(Repo `openclaw_voice_assist` auf den Pis). Erstes Ergebnis: `gaston`
(2026-07-05). Spec: `Wakeword_Studio_Spec.md` im Voice-Assist-Repo.

## Arbeitsverzeichnis

Läuft in `~/wakeword-studio/` (nicht im Repo: venv, voices/, Datensätze,
train_out/). Dieses Verzeichnis hier enthält die reproduzierbaren Skripte.

## Ablauf

1. **Setup**: `setup_downloads.sh` — venv-Deps, openwakeword (editable Clone),
   Basis-Modelle, ACAV100M-Features (~17 GB), Validierungs-Features.
   Dann `fix_downloads.py` (MIT-RIRs direkt vom MIT) und `fix_audioset.py`
   (AudioSet liegt auf HF inzwischen als Parquet, nicht mehr als tar!).
2. **Clips generieren**: `gen_clips.py` — Positives + Adversarial-Negatives
   mit deutschen Piper-Voices (piper-sample-generator, onnx-Voices,
   `--phoneme-input` für IPA). Stimm-Pool/Aussprachen oben im Skript.
3. **Trainieren**: `openwakeword/train.py --training_config gaston.yaml
   --augment_clips --overwrite` dann `--train_model`.
   VORHER: `podman stop llm` (lädt sonst on-demand und frisst beide GPUs),
   danach `podman start llm`.
4. **Evaluieren**: `eval_raw.py` (Recall je Stimme/Aussprache auf rohen
   Clips), `eval_sweep.py` (Threshold-Sweep), `eval_debounce.py`
   (FP/h mit 3-Frame-Debounce wie im Assistant — das ist die relevante Zahl).
5. **tflite**: `onnx2tf -i train_out/<name>.onnx -o out -kat x`
   (`-kat x` ist Pflicht, sonst wird der Input transponiert!).
6. **Deploy**: `<name>_float32.tflite` + manifest.yaml als Bundle nach
   `models/wakewords/<name>/` im Voice-Assist-Repo.

## Bekannte Stolpersteine (alle 2026-07-05 gelöst)

- **torchaudio >= 2.9** delegiert load/info an torchcodec (braucht
  FFmpeg-Libs, hier nicht vorhanden) → `openwakeword-kompat.patch`
  ersetzt sie in `openwakeword/data.py` durch soundfile.
- **scipy >= 1.15** bricht `acoustics` (sph_harm entfernt) → `scipy<1.15`.
- **torch 2.12 ONNX-Export** braucht `onnxscript`.
- **train.py erwartet alte piper-sample-generator-API** →
  `generate_samples_shim.py` ins piper-sample-generator-Repo-Root legen.
- **FMA-Dataset** ist mit datasets>=3 nicht mehr ladbar (Script-Dataset);
  AudioSet reicht als Background.
- Piper medium/high-Voices liefern 22.05 kHz — `resample_16k.py` falls
  Clips nicht ohnehin 16 kHz sind.

## Erkenntnisse Gaston-Training

- max_negative_weight 1500 + FP-Ziel 0.2/h → übervorsichtiges Modell
  (Recall 0.13). Besser: 500 / 0.5/h / layer_size 64.
- Schlechte TTS-Stimmen (kerstin gedrosselt, mls) erkennt das Modell
  nicht — vorher Hörproben machen und Pool kuratieren.
- FP-Bewertung immer MIT Debounce rechnen: roh 0.47 FP/h @0.35 sah
  schlecht aus, mit 3-Frame-Streak: 0.00 FP/h bis Threshold 0.25.
