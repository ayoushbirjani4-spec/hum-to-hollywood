"""Stage 1: load or record audio, convert to mono, trim silence, reduce noise."""
from __future__ import annotations

import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf

SR = 22050  # Basic Pitch works at 22.05 kHz


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    """Load any WAV/MP3/FLAC as mono float32 at `sr`."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def from_gradio(audio: tuple[int, np.ndarray], sr: int = SR) -> np.ndarray:
    """Convert Gradio's (sample_rate, int16/float array) tuple to mono float32 at `sr`."""
    in_sr, data = audio
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    if data.ndim == 2:
        data = data.mean(axis=1)
    if in_sr != sr:
        data = librosa.resample(data.astype(np.float32), orig_sr=in_sr, target_sr=sr)
    return data.astype(np.float32)


def clean_audio(y: np.ndarray, sr: int = SR, denoise: bool = True, top_db: float = 35) -> np.ndarray:
    """Trim leading/trailing silence, lightly denoise, peak-normalise."""
    if y.size == 0:
        raise ValueError("Audio is empty.")
    y, _ = librosa.effects.trim(y, top_db=top_db)
    if denoise and y.size > sr // 2:
        # stationary=True is gentle and fast; prop_decrease < 1 keeps the voice natural
        y = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.75)
    peak = np.max(np.abs(y)) if y.size else 0
    if peak < 1e-4:
        raise ValueError("Audio is silent. Try humming louder or closer to the mic.")
    return (0.95 * y / peak).astype(np.float32)


def save_wav(path: str, y: np.ndarray, sr: int = SR) -> str:
    sf.write(path, y, sr)
    return path
