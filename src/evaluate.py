"""Melody faithfulness: how much of the generated track follows the hummed tune.

For every moment where the guide melody plays a note, look at the generated
track's chromagram (energy per pitch class, scaled so the strongest is 1.0) and
read the value at the melody's pitch class. 100% means the melody note is always
the strongest pitch in the track; a track that ignores the tune scores about the
same as a random pitch class, which is reported alongside as the baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOP = 1024


@dataclass
class TuneMatch:
    score: float      # 0-1, mean chroma value at the melody's pitch class
    baseline: float   # 0-1, mean chroma value over all pitch classes (what ignoring the tune scores)
    lag: float        # seconds the track lags the guide at the best alignment


def _chroma(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    return librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=HOP)  # (12, frames), max-normalised


def melody_match(guide: np.ndarray, track: np.ndarray, sr: int, max_lag: float = 0.3) -> TuneMatch:
    n = min(guide.size, track.size)
    if n < sr // 2:
        return TuneMatch(0.0, 0.0, 0.0)
    g, t = _chroma(guide[:n], sr), _chroma(track[:n], sr)

    # frames where the guide actually plays a note (same framing as the chromagram)
    import librosa
    rms = librosa.feature.rms(y=guide[:n], frame_length=4096, hop_length=HOP)[0]
    m = min(rms.size, g.shape[1], t.shape[1])
    g, t, rms = g[:, :m], t[:, :m], rms[:m]
    active = rms > 0.1 * rms.max()
    pcs = g.argmax(axis=0)

    best, best_lag = -1.0, 0
    lag_frames = int(max_lag * sr / HOP)
    for lag in range(0, lag_frames + 1):  # the track may come in slightly late, never early
        idx = np.nonzero(active[: g.shape[1] - lag])[0]
        if idx.size == 0:
            continue
        s = float(t[pcs[idx], idx + lag].mean())
        if s > best:
            best, best_lag = s, lag
    if best < 0:
        return TuneMatch(0.0, 0.0, 0.0)
    return TuneMatch(score=best, baseline=float(t[:, active].mean()), lag=best_lag * HOP / sr)
