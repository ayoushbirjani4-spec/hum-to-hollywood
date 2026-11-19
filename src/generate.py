"""Stage 5: music generation with Meta MusicGen Melody.

MusicGen Melody takes a text prompt plus a melody (its chromagram) and generates
music that follows the melody's shape. Model weights are CC-BY-NC 4.0: fine for a
portfolio, not for commercial use.

Backends (set with the HTH_BACKEND environment variable):
  musicgen  real model via Hugging Face transformers (needs a GPU to be practical)
  mock      instant placeholder: guide melody over a simple beat. For testing the
            app on a laptop without a GPU. Output is NOT AI-generated music.
"""
from __future__ import annotations

import os

import numpy as np

MODEL_ID = os.environ.get("HTH_MODEL", "facebook/musicgen-melody")
SAMPLE_RATE = 32000
TOKENS_PER_SECOND = 50
MAX_SECONDS = 30

_model = None
_processor = None


def backend() -> str:
    b = os.environ.get("HTH_BACKEND")
    if b:
        return b
    try:
        import torch
        return "musicgen" if torch.cuda.is_available() else "mock"
    except ImportError:
        return "mock"


def _load():
    global _model, _processor
    if _model is None:
        import torch
        from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = MusicgenMelodyForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
        _model.eval()
    return _model, _processor


def generate(prompt: str, guide: np.ndarray, sr: int = SAMPLE_RATE, duration: float = 15.0,
             guidance_scale: float = 3.0, bpm: float = 100.0, seed: int | None = None) -> tuple[np.ndarray, int]:
    """Return (audio mono float32, sample_rate) following `guide` in the style of `prompt`."""
    duration = float(np.clip(duration, 2, MAX_SECONDS))
    if backend() == "mock":
        return _mock(guide, sr, duration, bpm), sr

    import torch
    model, processor = _load()
    if sr != SAMPLE_RATE:
        import librosa
        guide = librosa.resample(guide, orig_sr=sr, target_sr=SAMPLE_RATE)
    if seed is not None:
        torch.manual_seed(seed)

    inputs = processor(audio=guide, sampling_rate=SAMPLE_RATE, text=[prompt], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(model.dtype)
    with torch.inference_mode():
        audio = model.generate(**inputs, do_sample=True, guidance_scale=guidance_scale,
                               max_new_tokens=int(duration * TOKENS_PER_SECOND))
    out = audio[0, 0].float().cpu().numpy()
    return out.astype(np.float32), model.config.audio_encoder.sampling_rate


def _mock(guide: np.ndarray, sr: int, duration: float, bpm: float) -> np.ndarray:
    """Loop the guide melody to `duration` over a kick/hat pattern. Placeholder only."""
    n = int(duration * sr)
    reps = int(np.ceil(n / max(guide.size, 1)))
    out = np.tile(guide, reps)[:n] * 0.6
    beat = int(sr * 60 / bpm)
    t = np.arange(int(0.12 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-t * 40)) * t) * np.exp(-t * 25)
    hat = np.random.default_rng(0).standard_normal(int(0.03 * sr)) * np.exp(-np.arange(int(0.03 * sr)) / (0.006 * sr))
    for i, pos in enumerate(range(0, n, beat // 2)):
        src = kick if i % 2 == 0 else hat * 0.15
        end = min(n, pos + src.size)
        out[pos:end] += src[: end - pos] * 0.5
    return out.astype(np.float32)
