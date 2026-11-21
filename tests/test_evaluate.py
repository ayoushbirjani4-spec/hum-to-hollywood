import numpy as np

from src.evaluate import melody_match
from src.melody import render_guide
from src.transcribe import Note

SR = 32000
TUNE = [60, 62, 64, 65, 67, 65, 64, 62]


def notes(pitches, shift=0, dur=0.5):
    return [Note(p + shift, i * dur, (i + 1) * dur, 100) for i, p in enumerate(pitches)]


def test_identical_track_scores_near_perfect():
    g = render_guide(notes(TUNE), SR)
    m = melody_match(g, g, SR)
    assert m.score > 0.95 and m.lag == 0


def test_different_tune_scores_low():
    g = render_guide(notes(TUNE), SR)
    other = render_guide(notes(list(reversed(TUNE)), shift=1), SR)
    m = melody_match(g, other, SR)
    assert m.score < 0.5


def test_tune_under_noise_beats_baseline():
    rng = np.random.default_rng(0)
    g = render_guide(notes(TUNE), SR)
    track = 0.5 * g + 0.3 * rng.standard_normal(g.size).astype(np.float32)
    m = melody_match(g, track, SR)
    assert m.score > m.baseline + 0.2


def test_odd_lengths_do_not_crash():
    g = render_guide(notes(TUNE), SR)
    for extra in (0, 1, 513, 1023, 1024, 1025):
        x = np.concatenate([g, np.zeros(extra, dtype=np.float32)])
        assert melody_match(x, x, SR).score > 0.9


def test_late_track_is_aligned():
    g = render_guide(notes(TUNE), SR)
    late = np.concatenate([np.zeros(int(0.2 * SR), dtype=np.float32), g])
    m = melody_match(g, late, SR)
    assert m.score > 0.9 and 0.1 < m.lag < 0.3
