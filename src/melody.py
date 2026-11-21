"""Stage 3: melody clean-up.

Raw Basic Pitch output from a hum is noisy: stray blips, overlapping notes,
wobbly pitch that splits one sung note into several, off-key notes, and loose
timing. This module turns that into a clean, monophonic, in-key, quantised melody.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .transcribe import Note

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Krumhansl-Kessler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10, 11)  # natural minor + raised 7th (harmonic minor leading tone)


@dataclass
class Key:
    tonic: int   # pitch class 0-11
    mode: str    # "major" or "minor"

    @property
    def name(self) -> str:
        return f"{NOTE_NAMES[self.tonic]} {self.mode}"

    @property
    def pitch_classes(self) -> set[int]:
        scale = MAJOR_SCALE if self.mode == "major" else MINOR_SCALE
        return {(self.tonic + s) % 12 for s in scale}


@dataclass
class CleanMelody:
    notes: list[Note]
    key: Key
    bpm: float


# ---------- individual steps ----------

def remove_strays(notes: list[Note], min_dur: float = 0.06, min_velocity: int = 20) -> list[Note]:
    """Drop blips that are too short or too quiet to be intended notes."""
    return [n for n in notes if n.duration >= min_dur and n.velocity >= min_velocity]


def make_monophonic(notes: list[Note]) -> list[Note]:
    """A hum is one voice. Where notes overlap, keep the stronger one and trim the other."""
    out: list[Note] = []
    for n in sorted(notes, key=lambda n: (n.start, -n.velocity)):
        n = replace(n)
        if out and n.start < out[-1].end:
            prev = out[-1]
            # Score = loudness weighted by length: harmonics/ghosts are short and quiet
            if n.velocity * n.duration > prev.velocity * prev.duration:
                prev.end = n.start
                if prev.duration < 0.03:
                    out.pop()
            else:
                n.start = prev.end
                if n.duration < 0.03:
                    continue
        out.append(n)
    return out


def detect_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    """Times (s) where the hum is re-articulated: energy jumps, not pitch changes."""
    import librosa
    return librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)


def merge_wobbles(notes: list[Note], onsets: np.ndarray | None = None, max_gap: float = 0.06,
                  wobble_dur: float = 0.12, onset_tol: float = 0.06) -> list[Note]:
    """Merge a held note that was split into pieces by vibrato or pitch drift.

    - Same pitch, small gap, and no audio onset at the boundary -> one note.
      (Basic Pitch reports a vibrato split and a genuinely repeated note the same
      way, so the audio onset is what tells them apart. Without onsets, only
      near-zero gaps are merged.)
    - A very short note within 1 semitone of its neighbour -> absorbed into the neighbour.
    """
    if not notes:
        return []

    def reattacked(t: float) -> bool:
        return onsets is not None and onsets.size > 0 and np.min(np.abs(onsets - t)) <= onset_tol

    out = [replace(notes[0])]
    for n in notes[1:]:
        prev = out[-1]
        gap = n.start - prev.end
        same = n.pitch == prev.pitch and gap <= (max_gap if onsets is not None else 0.02) and not reattacked(n.start)
        wobble = (gap <= max_gap and abs(n.pitch - prev.pitch) <= 1
                  and min(n.duration, prev.duration) < wobble_dur)
        if same or wobble:
            # keep the pitch of the longer piece
            if n.duration > prev.duration:
                prev.pitch = n.pitch
            prev.end = max(prev.end, n.end)
            prev.velocity = max(prev.velocity, n.velocity)
        else:
            out.append(replace(n))
    return out


def detect_key(notes: list[Note]) -> Key:
    """Krumhansl-Schmuckler key finding on a duration-weighted pitch-class histogram."""
    hist = np.zeros(12)
    for n in notes:
        hist[n.pitch % 12] += n.duration
    if hist.sum() == 0:
        return Key(0, "major")
    best, best_r = Key(0, "major"), -np.inf
    for tonic in range(12):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            r = np.corrcoef(hist, np.roll(profile, tonic))[0, 1]
            if r > best_r:
                best, best_r = Key(tonic, mode), r
    return best


def snap_to_key(notes: list[Note], key: Key) -> list[Note]:
    """Move each off-key note one semitone to the in-key neighbour closest to the previous note."""
    allowed = key.pitch_classes
    out: list[Note] = []
    for n in notes:
        n = replace(n)
        if n.pitch % 12 not in allowed:
            up, down = n.pitch + 1, n.pitch - 1
            cands = [p for p in (down, up) if p % 12 in allowed]
            if len(cands) == 2 and out:
                ref = out[-1].pitch
                n.pitch = min(cands, key=lambda p: (abs(p - ref), p))
            else:
                n.pitch = cands[0]
        out.append(n)
    return out


def estimate_tempo(notes: list[Note], y: np.ndarray | None = None, sr: int = 22050) -> float:
    """Estimate tempo from the hum audio (onset strength); fall back to note spacing."""
    bpm = 0.0
    if y is not None and y.size > sr:
        import librosa
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        bpm = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset_env, sr=sr, start_bpm=100))[0])
    if not bpm and len(notes) > 2:
        iois = np.diff([n.start for n in notes])
        iois = iois[iois > 0.1]
        if iois.size:
            bpm = 60.0 / float(np.median(iois))
    bpm = bpm or 100.0
    # fold octave errors into a musical range
    while bpm < 70:
        bpm *= 2
    while bpm > 160:
        bpm /= 2
    return round(bpm, 1)


def quantize(notes: list[Note], bpm: float, division: int = 4) -> list[Note]:
    """Snap note starts/ends to a grid (division=4 -> 16th notes). Melody starts at t=0."""
    if not notes:
        return []
    grid = 60.0 / bpm / division
    t0 = notes[0].start
    q: list[Note] = []
    for n in notes:
        s = round((n.start - t0) / grid) * grid
        e = max(s + grid, round((n.end - t0) / grid) * grid)
        if q and abs(s - q[-1].start) < 1e-9:
            # two notes landed on the same grid slot: keep the longer
            if n.duration > q[-1].duration:
                q[-1] = Note(n.pitch, s, e, n.velocity)
            continue
        q.append(Note(n.pitch, s, e, n.velocity))
    for a, b in zip(q, q[1:]):
        if a.end > b.start:
            a.end = b.start
    return q


# ---------- full clean-up ----------

def clean_melody(raw: list[Note], y: np.ndarray | None = None, sr: int = 22050,
                 bpm: float | None = None, snap: bool = True, quantise: bool = True) -> CleanMelody:
    onsets = detect_onsets(y, sr) if y is not None else None
    notes = remove_strays(raw)
    notes = make_monophonic(notes)
    notes = merge_wobbles(notes, onsets)
    if not notes:
        raise ValueError("No clear notes found in the hum. Try a slower, steadier hum.")
    key = detect_key(notes)
    if snap:
        notes = merge_wobbles(snap_to_key(notes, key), onsets)
    bpm = bpm or estimate_tempo(notes, y, sr)
    if quantise:
        notes = quantize(notes, bpm)
    else:
        t0 = notes[0].start
        notes = [Note(n.pitch, n.start - t0, n.end - t0, n.velocity) for n in notes]
    return CleanMelody(notes=notes, key=key, bpm=bpm)


# ---------- outputs: MIDI, guide audio, piano roll ----------

def to_midi(melody: CleanMelody, path: str, program: int = 73) -> str:
    """Write the cleaned melody as a MIDI file (default instrument: flute)."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=melody.bpm)
    inst = pretty_midi.Instrument(program=program, name="Melody")
    for n in melody.notes:
        inst.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end))
    pm.instruments.append(inst)
    pm.write(path)
    return path


def loop_notes(notes: list[Note], bpm: float, duration: float, beats_per_bar: int = 4) -> list[Note]:
    """Repeat the melody, in whole bars, until it covers `duration` seconds.

    MusicGen only follows the melody for as long as the guide lasts; a 7 s hum
    guiding a 15 s track leaves the second half unguided.
    """
    if not notes:
        return []
    bar = 60.0 / bpm * beats_per_bar
    phrase = max(bar, np.ceil(notes[-1].end / bar - 1e-6) * bar)  # round up to whole bars
    out: list[Note] = []
    offset = 0.0
    while offset < duration:
        out.extend(Note(n.pitch, n.start + offset, n.end + offset, n.velocity)
                   for n in notes if n.start + offset < duration)
        offset += phrase
    return out


# Octave harmonics only (1, 2, 4 x f0): they all map to the same pitch class, so
# MusicGen's chromagram sees a clean single note. A 3rd harmonic would add a
# phantom fifth above every note.
_GUIDE_HARMONICS = ((1, 1.0), (2, 0.5), (4, 0.25))


def render_guide(notes: list[Note], sr: int = 32000, tail: float = 0.5) -> np.ndarray:
    """Synthesize a clean guide track for MusicGen's melody conditioning."""
    if not notes:
        return np.zeros(sr, dtype=np.float32)
    total = int((notes[-1].end + tail) * sr)
    out = np.zeros(total, dtype=np.float32)
    for n in notes:
        f0 = 440.0 * 2 ** ((n.pitch - 69) / 12)
        s, e = int(n.start * sr), int(n.end * sr)
        t = np.arange(e - s) / sr
        tone = sum(a * np.sin(2 * np.pi * f0 * k * t) for k, a in _GUIDE_HARMONICS)
        env = np.ones_like(t)
        a, r = min(int(0.01 * sr), t.size // 4), min(int(0.06 * sr), t.size // 3)
        if a:
            env[:a] = np.linspace(0, 1, a)
        if r:
            env[-r:] = np.linspace(1, 0, r)
        out[s:e] += (tone * env * (n.velocity / 127)).astype(np.float32)
    peak = np.max(np.abs(out))
    return out / peak * 0.8 if peak > 0 else out


def piano_roll(raw: list[Note], melody: CleanMelody):
    """Matplotlib figure: raw detected notes (grey) vs cleaned melody (colour)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=110)
    if raw:
        t0 = raw[0].start
        for n in raw:
            ax.add_patch(plt.Rectangle((n.start - t0, n.pitch - 0.4), n.duration, 0.8,
                                       color="#b8b8b8", alpha=0.6, lw=0))
    for n in melody.notes:
        ax.add_patch(plt.Rectangle((n.start, n.pitch - 0.4), n.duration, 0.8,
                                   facecolor="#c2410c", edgecolor="white", lw=1.2))
    pitches = [n.pitch for n in (raw or []) + melody.notes]
    ax.set_ylim(min(pitches) - 2, max(pitches) + 2)
    end = max([n.end for n in melody.notes] + [n.end - (raw[0].start if raw else 0) for n in raw])
    ax.set_xlim(0, end + 0.2)
    ticks = range(min(pitches) - 1, max(pitches) + 2)
    ax.set_yticks([p for p in ticks if p % 12 in (0, 4, 7)])
    ax.set_yticklabels([f"{NOTE_NAMES[p % 12]}{p // 12 - 1}" for p in ticks if p % 12 in (0, 4, 7)])
    ax.set_xlabel("seconds")
    ax.set_title(f"Detected melody  ·  {melody.key.name}  ·  {melody.bpm:.0f} bpm   (grey = raw, orange = cleaned)",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig
