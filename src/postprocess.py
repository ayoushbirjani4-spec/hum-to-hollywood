"""Stage 6: loudness, fades, optional hum overlay, export."""
from __future__ import annotations

import numpy as np
import soundfile as sf

TARGET_LUFS = -14.0  # streaming-platform loudness


def normalize_loudness(y: np.ndarray, sr: int, target: float = TARGET_LUFS) -> np.ndarray:
    import pyloudnorm as pyln
    if y.size < int(0.4 * sr):  # pyloudnorm needs at least one 400 ms block
        return y
    meter = pyln.Meter(sr)
    loud = meter.integrated_loudness(y)
    if not np.isfinite(loud):
        return y
    import warnings
    with warnings.catch_warnings():
        # pyloudnorm warns when the gain pushes peaks past 1.0; the peak scaling below handles that
        warnings.filterwarnings("ignore", message="Possible clipped samples")
        y = pyln.normalize.loudness(y, loud, target)
    peak = np.max(np.abs(y))
    return y / peak * 0.98 if peak > 0.98 else y  # simple peak safety


def fade(y: np.ndarray, sr: int, fade_in: float = 0.05, fade_out: float = 1.5) -> np.ndarray:
    y = y.copy()
    a, b = min(int(fade_in * sr), y.size // 2), min(int(fade_out * sr), y.size // 2)
    if a:
        y[:a] *= np.linspace(0, 1, a)
    if b:
        y[-b:] *= np.linspace(1, 0, b)
    return y


def mix_hum(track: np.ndarray, sr: int, hum: np.ndarray, hum_sr: int, level: float = 0.25) -> np.ndarray:
    """Lay the user's original hum softly over the start of the track."""
    import librosa
    if hum_sr != sr:
        hum = librosa.resample(hum, orig_sr=hum_sr, target_sr=sr)
    hum = hum[: track.size]
    out = track.copy()
    rms_t = np.sqrt(np.mean(track ** 2)) + 1e-9
    rms_h = np.sqrt(np.mean(hum ** 2)) + 1e-9
    out[: hum.size] += hum * (rms_t / rms_h) * level
    return out


def finish(y: np.ndarray, sr: int, hum: np.ndarray | None = None, hum_sr: int | None = None,
           hum_level: float = 0.0) -> np.ndarray:
    if hum is not None and hum_level > 0:
        y = mix_hum(y, sr, hum, hum_sr or sr, hum_level)
    return fade(normalize_loudness(y, sr), sr).astype(np.float32)


def export(y: np.ndarray, sr: int, stem: str) -> dict[str, str]:
    """Write <stem>.wav and <stem>.mp3 (libsndfile >= 1.1 writes MP3, no ffmpeg needed)."""
    paths = {"wav": f"{stem}.wav", "mp3": f"{stem}.mp3"}
    sf.write(paths["wav"], y, sr)
    try:
        sf.write(paths["mp3"], y, sr, format="MP3")
    except Exception:
        paths.pop("mp3")
    return paths
