"""Stage 4: style presets -> rich text prompts for MusicGen."""
from __future__ import annotations

PRESETS: dict[str, str] = {
    "Epic film score": "epic orchestral film score, soaring strings, bold brass, big taiko drums, choir, cinematic, heroic",
    "Emotional film score": "emotional cinematic score, solo piano, warm string ensemble, gentle swells, intimate, hopeful",
    "Dark thriller": "dark tense thriller soundtrack, low strings ostinato, pulsing synth bass, ticking percussion, suspense",
    "Pop": "modern upbeat pop, bright synths, punchy drums, catchy, polished radio production",
    "Rock": "energetic rock, distorted electric guitars, driving bass, live drums, anthemic",
    "Lo-fi": "lo-fi hip hop, mellow rhodes piano, vinyl crackle, dusty drums, relaxed, chill",
    "Synthwave": "80s synthwave, retro analog synths, gated reverb drums, arpeggiated bass, neon night drive",
    "Jazz": "smooth jazz combo, saxophone lead, upright bass, brushed drums, piano comping, late night club",
}


def build_prompt(style: str, bpm: float | None = None, key_name: str | None = None, extra: str = "") -> str:
    """Combine a preset (or free text) with tempo and key from the melody clean-up."""
    base = PRESETS.get(style, style).strip()
    parts = [base]
    if extra.strip():
        parts.append(extra.strip())
    if bpm:
        parts.append(f"{round(bpm)} bpm")
    if key_name:
        parts.append(f"in {key_name}")
    return ", ".join(parts)
