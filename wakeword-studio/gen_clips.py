#!/usr/bin/env python3
"""Generiert deutsche Trainings-Clips für das Gaston-Wakeword (Meilenstein 2).

Stimm-Pool und Aussprache-Kandidaten stammen aus Jochens Hörproben-Feedback
(Runde 1+2): Nasal ɡastˈɔ̃ ist Favorit (60%), Gastong zweiter (40%).
Läuft auf dem ai-stack, Ausgabe direkt ins train.py-Layout:
train_out/gaston/{positive_train,positive_test,negative_train,negative_test}
"""
import multiprocessing as mp
import os
import shutil
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "3")

HOME = os.path.expanduser("~/wakeword-studio")
sys.path.insert(0, os.path.join(HOME, "piper-sample-generator"))

VOICES = os.path.join(HOME, "voices")
OUT = os.path.join(HOME, "train_out", "gaston")
TMP = os.path.join(HOME, "train_out", "tmp")

N_POS_TRAIN = 30000
N_POS_TEST = 3000
N_NEG_TRAIN = 30000
N_NEG_TEST = 3000

NOISE = dict(noise_scales=(0.333, 0.667, 1.0), noise_scale_ws=(0.6, 0.8, 1.0))


def V(name: str) -> str:
    return os.path.join(VOICES, f"de_DE-{name}.onnx")


# (voice, length_scales) — Einschränkungen aus dem Hörproben-Feedback
NASAL_POOL = [
    ("thorsten-medium", (0.8, 0.9, 1.0, 1.1, 1.2)),
    ("thorsten-high", (0.8, 0.9, 1.0, 1.1, 1.2)),
    ("thorsten_emotional-medium", (0.8, 0.9, 1.0, 1.1, 1.2)),
    ("karlsson-low", (0.8, 0.9, 1.0, 1.1, 1.2)),
    ("pavoque-low", (0.8, 0.9, 1.0, 1.1, 1.2)),
    ("ramona-low", (1.0, 1.1, 1.2, 1.3)),      # unter 1.0 unbrauchbar
    ("kerstin-low", (1.2, 1.3, 1.4, 1.5)),     # nur stark gedrosselt
]
GASTONG_POOL = [
    ("thorsten-medium", (0.8, 0.9, 1.0, 1.1, 1.2)),
    ("thorsten-high", (0.8, 0.9, 1.0, 1.1, 1.2)),
    ("thorsten_emotional-medium", (0.8, 0.9, 1.0, 1.1, 1.2)),
    ("karlsson-low", (0.8, 0.9, 1.0, 1.1, 1.2)),
    ("pavoque-low", (0.8, 0.9, 1.0, 1.1, 1.2)),
    ("eva_k-x_low", (0.9, 1.0, 1.1, 1.2)),     # für Gastong ok, für Nasal nicht
    ("ramona-low", (1.0, 1.1, 1.2, 1.3)),
]
# kerstin fällt bei Gastong komplett raus (Feedback: unbrauchbar)

# Adversarial-Negatives: phonetisch ähnliche deutsche Wörter/Phrasen
NEG_TEXTS = [
    "Bastion", "Station", "Tankstelle", "Gastank", "Gastro", "Gasthof",
    "Gasthaus", "Gaststätte", "Gastgeber", "Kasten", "Kastanie", "Gasse",
    "Gassi", "Gast", "Gäste", "gestern", "Karton", "Balkon", "Beton",
    "Salon", "Saison", "Waggon", "Chanson", "Marathon", "Pastor", "Gas",
    "Gasflasche", "der Gast ist da", "auf dem Balkon", "die Gäste kommen",
    "mach das Gas an", "im Salon sitzen",
]
NEG_VOICES = [
    ("thorsten-medium", (0.9, 1.0, 1.1)),
    ("thorsten-high", (0.9, 1.0, 1.1)),
    ("thorsten_emotional-medium", (0.9, 1.0, 1.1)),
    ("karlsson-low", (0.9, 1.0, 1.1)),
    ("pavoque-low", (0.9, 1.0, 1.1)),
    ("eva_k-x_low", (0.9, 1.0, 1.1)),
    ("ramona-low", (1.0, 1.1, 1.2)),
    ("kerstin-low", (1.1, 1.2, 1.3)),
]


def build_jobs():
    jobs = []
    for split, n_pos, n_neg in (("train", N_POS_TRAIN, N_NEG_TRAIN),
                                ("test", N_POS_TEST, N_NEG_TEST)):
        n_nasal = int(n_pos * 0.6)
        n_gastong = n_pos - n_nasal
        for pool, total, text, phon in (
            (NASAL_POOL, n_nasal, "ɡastˈɔ̃", True),
            (GASTONG_POOL, n_gastong, "Gastong", False),
        ):
            per_voice = total // len(pool)
            for voice, ls in pool:
                tag = f"{'nasal' if phon else 'gastong'}_{voice}_{split}"
                jobs.append(dict(
                    tag=tag,
                    dest=os.path.join(OUT, f"positive_{split}"),
                    text=text, phoneme_input=phon, model=V(voice),
                    max_samples=per_voice, length_scales=ls,
                ))
        per_voice = n_neg // len(NEG_VOICES)
        for voice, ls in NEG_VOICES:
            tag = f"neg_{voice}_{split}"
            jobs.append(dict(
                tag=tag,
                dest=os.path.join(OUT, f"negative_{split}"),
                text=NEG_TEXTS, phoneme_input=False, model=V(voice),
                max_samples=per_voice, length_scales=ls,
            ))
    return jobs


def run_job(job):
    from piper_sample_generator.__main__ import generate_samples_onnx
    t0 = time.time()
    tmpdir = os.path.join(TMP, job["tag"])
    os.makedirs(tmpdir, exist_ok=True)
    os.makedirs(job["dest"], exist_ok=True)
    generate_samples_onnx(
        text=job["text"],
        output_dir=tmpdir,
        model=job["model"],
        max_samples=job["max_samples"],
        phoneme_input=job["phoneme_input"],
        length_scales=job["length_scales"],
        **NOISE,
    )
    n = 0
    for f in os.listdir(tmpdir):
        if f.endswith(".wav"):
            shutil.move(os.path.join(tmpdir, f),
                        os.path.join(job["dest"], f"{job['tag']}_{f}"))
            n += 1
    shutil.rmtree(tmpdir, ignore_errors=True)
    return job["tag"], n, time.time() - t0


def main():
    jobs = build_jobs()
    total = sum(j["max_samples"] for j in jobs)
    print(f"{len(jobs)} Jobs, Soll gesamt: {total} Clips", flush=True)
    done = 0
    with mp.Pool(4) as pool:
        for tag, n, dt in pool.imap_unordered(run_job, jobs):
            done += n
            print(f"[{done}/{total}] {tag}: {n} Clips in {dt:.0f}s", flush=True)
    print("FERTIG", flush=True)
    for d in ("positive_train", "positive_test", "negative_train", "negative_test"):
        p = os.path.join(OUT, d)
        print(d, len([f for f in os.listdir(p) if f.endswith('.wav')]), flush=True)


if __name__ == "__main__":
    main()
