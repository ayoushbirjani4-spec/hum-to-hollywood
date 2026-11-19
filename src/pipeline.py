"""Runs every stage in order: hum in, arranged track out.

CLI:
    python -m src.pipeline my_hum.wav --style "Epic film score" --duration 15
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import audio_in, generate, melody, postprocess, prompts, transcribe

OUT_DIR = os.environ.get("HTH_OUT", "outputs")


@dataclass
class Result:
    prompt: str
    key: str
    bpm: float
    n_raw_notes: int
    n_notes: int
    files: dict[str, str] = field(default_factory=dict)  # track_wav, track_mp3, midi, guide_wav, piano_roll
    backend: str = ""
    seconds: dict[str, float] = field(default_factory=dict)


def hum_to_melody(y: np.ndarray, sr: int = audio_in.SR, bpm: float | None = None):
    """Stages 1-3 only. Returns (raw_notes, clean_melody, cleaned_audio)."""
    y = audio_in.clean_audio(y, sr)
    raw = transcribe.transcribe(y, sr)
    clean = melody.clean_melody(raw, y, sr, bpm=bpm)
    return raw, clean, y


def run(y: np.ndarray, sr: int = audio_in.SR, style: str = "Epic film score", extra: str = "",
        duration: float = 15.0, bpm: float | None = None, hum_level: float = 0.0,
        seed: int | None = None, out_dir: str = OUT_DIR) -> Result:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, time.strftime("%Y%m%d-%H%M%S"))
    t = {}

    t0 = time.perf_counter()
    raw, clean, hum = hum_to_melody(y, sr, bpm)
    t["melody"] = time.perf_counter() - t0

    files = {"midi": melody.to_midi(clean, f"{stem}_melody.mid")}
    guide = melody.render_guide(clean.notes, sr=generate.SAMPLE_RATE)
    files["guide_wav"] = audio_in.save_wav(f"{stem}_guide.wav", guide, generate.SAMPLE_RATE)
    fig = melody.piano_roll(raw, clean)
    files["piano_roll"] = f"{stem}_pianoroll.png"
    fig.savefig(files["piano_roll"])
    import matplotlib.pyplot as plt
    plt.close(fig)

    prompt = prompts.build_prompt(style, clean.bpm, clean.key.name, extra)

    t0 = time.perf_counter()
    track, track_sr = generate.generate(prompt, guide, generate.SAMPLE_RATE, duration, bpm=clean.bpm, seed=seed)
    t["generate"] = time.perf_counter() - t0

    track = postprocess.finish(track, track_sr, hum, sr, hum_level)
    for fmt, path in postprocess.export(track, track_sr, f"{stem}_track").items():
        files[f"track_{fmt}"] = path

    return Result(prompt=prompt, key=clean.key.name, bpm=clean.bpm, n_raw_notes=len(raw),
                  n_notes=len(clean.notes), files=files, backend=generate.backend(), seconds=t)


def main():
    ap = argparse.ArgumentParser(description="Hum to Hollywood")
    ap.add_argument("audio", help="WAV/MP3 of you humming")
    ap.add_argument("--style", default="Epic film score", help=f"preset or free text. Presets: {', '.join(prompts.PRESETS)}")
    ap.add_argument("--extra", default="", help="extra words for the prompt")
    ap.add_argument("--duration", type=float, default=15)
    ap.add_argument("--bpm", type=float, default=None, help="override detected tempo")
    ap.add_argument("--hum-level", type=float, default=0.0, help="0-1, mix your hum on top")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    y = audio_in.load_audio(a.audio)
    r = run(y, style=a.style, extra=a.extra, duration=a.duration, bpm=a.bpm, hum_level=a.hum_level, seed=a.seed)
    print(f"backend : {r.backend}")
    print(f"melody  : {r.n_raw_notes} raw notes -> {r.n_notes} clean notes, {r.key}, {r.bpm:.0f} bpm")
    print(f"prompt  : {r.prompt}")
    print(f"timing  : " + ", ".join(f"{k} {v:.1f}s" for k, v in r.seconds.items()))
    for k, v in r.files.items():
        print(f"{k:<10}: {v}")


if __name__ == "__main__":
    main()
