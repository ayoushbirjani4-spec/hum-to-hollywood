"""End-to-end on a synthetic 'hum' (voice-like tone with vibrato and noise). Uses the mock backend."""
import os

import numpy as np
import pytest

os.environ["HTH_BACKEND"] = "mock"

from src import pipeline  # noqa: E402

SR = 22050
MELODY = [60, 62, 64, 65, 67, 67, 65, 64, 62, 60]  # C D E F G G F E D C


def fake_hum(pitches, note_dur=0.5, sr=SR, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for p in pitches:
        t = np.arange(int(note_dur * sr)) / sr
        f = 440 * 2 ** ((p - 69) / 12) * (1 + 0.006 * np.sin(2 * np.pi * 5.5 * t))  # vibrato
        phase = 2 * np.pi * np.cumsum(f) / sr
        tone = np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
        env = np.minimum(1, np.minimum(t / 0.03, (note_dur - t) / 0.05))
        out.append(tone * env)
        out.append(np.zeros(int(0.03 * sr)))
    y = np.concatenate([np.zeros(sr // 2), *out, np.zeros(sr // 2)])
    return (0.4 * y + 0.01 * rng.standard_normal(y.size)).astype(np.float32)


@pytest.fixture(scope="module")
def melody_result():
    return pipeline.hum_to_melody(fake_hum(MELODY), SR, bpm=None)


def test_transcription_recovers_melody(melody_result):
    _, clean, _ = melody_result
    got = [n.pitch for n in clean.notes]
    # allow octave errors from the pitch tracker, compare pitch classes
    assert [p % 12 for p in got] == [p % 12 for p in MELODY], got
    assert clean.key.name == "C major"


def test_full_run_writes_files(tmp_path):
    r = pipeline.run(fake_hum(MELODY[:6]), SR, style="Pop", duration=4, out_dir=str(tmp_path))
    for k in ("track_wav", "midi", "guide_wav", "piano_roll"):
        assert os.path.getsize(r.files[k]) > 0, k
    assert "bpm" in r.prompt and "C major" in r.prompt
