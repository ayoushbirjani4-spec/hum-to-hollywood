"""Stage 2: hum -> notes with Spotify Basic Pitch."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np

from .audio_in import SR, save_wav


@dataclass
class Note:
    pitch: int        # MIDI note number
    start: float      # seconds
    end: float        # seconds
    velocity: int = 90

    @property
    def duration(self) -> float:
        return self.end - self.start


# Human humming range, roughly E2 to C6
MIN_FREQ = 80.0
MAX_FREQ = 1050.0

_model = None


def _get_model():
    global _model
    if _model is None:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model
        _model = Model(ICASSP_2022_MODEL_PATH)
    return _model


def transcribe(y: np.ndarray, sr: int = SR) -> list[Note]:
    """Run Basic Pitch on a mono signal and return raw notes (may overlap / be noisy)."""
    from basic_pitch.inference import predict

    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        save_wav(tmp, y, sr)
        _, _, note_events = predict(
            tmp,
            _get_model(),
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=58,  # ms
            minimum_frequency=MIN_FREQ,
            maximum_frequency=MAX_FREQ,
            melodia_trick=True,
        )
    finally:
        os.remove(tmp)

    notes = [
        Note(pitch=int(p), start=float(s), end=float(e), velocity=int(np.clip(a * 127, 1, 127)))
        for s, e, p, a, *_ in note_events
    ]
    return sorted(notes, key=lambda n: n.start)
