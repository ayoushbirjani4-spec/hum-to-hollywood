import numpy as np

from src.melody import (Key, clean_melody, detect_key, loop_notes, make_monophonic, merge_wobbles,
                        quantize, remove_strays, render_guide, snap_to_key)
from src.transcribe import Note


def seq(pitches, dur=0.5, gap=0.0):
    t, out = 0.0, []
    for p in pitches:
        out.append(Note(p, t, t + dur, 90))
        t += dur + gap
    return out


def test_remove_strays_drops_blips_and_quiet():
    notes = [Note(60, 0, 0.5, 90), Note(62, 0.5, 0.52, 90), Note(64, 0.6, 1.0, 5)]
    assert [n.pitch for n in remove_strays(notes)] == [60]


def test_monophonic_keeps_stronger_overlap():
    notes = [Note(60, 0, 1.0, 100), Note(72, 0.2, 0.4, 30)]  # quiet octave ghost
    out = make_monophonic(notes)
    assert [n.pitch for n in out] == [60]


def test_merge_wobbles_joins_split_held_note():
    notes = [Note(60, 0, 0.4, 90), Note(61, 0.42, 0.5, 70), Note(60, 0.51, 1.0, 90)]
    out = merge_wobbles(notes, onsets=np.array([0.0]))
    assert len(out) == 1 and out[0].pitch == 60 and out[0].end == 1.0


def test_merge_keeps_reattacked_repeat():
    notes = [Note(67, 0, 0.5, 90), Note(67, 0.5, 1.0, 90)]
    assert len(merge_wobbles(notes, onsets=np.array([0.0, 0.51]))) == 2
    assert len(merge_wobbles(notes, onsets=np.array([0.0]))) == 1


def test_merge_keeps_real_steps():
    out = merge_wobbles(seq([60, 62, 64]))
    assert [n.pitch for n in out] == [60, 62, 64]


def test_detect_key_major_and_minor():
    c_major = seq([60, 62, 64, 65, 67, 69, 71, 72, 67, 64, 60, 60])
    assert detect_key(c_major).name == "C major"
    a_minor = seq([57, 59, 60, 62, 64, 65, 67, 69, 64, 60, 57, 57])
    assert detect_key(a_minor).name == "A minor"


def test_snap_to_key_moves_off_key_notes():
    out = snap_to_key(seq([60, 61, 64, 66]), Key(0, "major"))
    assert all(n.pitch % 12 in Key(0, "major").pitch_classes for n in out)


def test_quantize_aligns_to_grid():
    bpm = 120  # 16th = 0.125 s
    notes = [Note(60, 1.0, 1.24, 90), Note(62, 1.26, 1.51, 90), Note(64, 1.49, 2.0, 90)]
    q = quantize(notes, bpm)
    assert q[0].start == 0.0
    for n in q:
        assert abs(n.start / 0.125 - round(n.start / 0.125)) < 1e-6
    assert all(a.end <= b.start for a, b in zip(q, q[1:]))


def test_clean_melody_end_to_end_on_notes():
    raw = seq([60, 62, 64, 65, 67, 65, 64, 62, 60], dur=0.48, gap=0.02)
    raw.insert(3, Note(80, 1.1, 1.13, 40))  # stray blip
    m = clean_melody(raw, bpm=120)
    assert [n.pitch for n in m.notes] == [60, 62, 64, 65, 67, 65, 64, 62, 60]
    assert m.key.name == "C major"


def test_loop_notes_fills_duration_in_whole_bars():
    notes = seq([60, 62, 64], dur=0.5)          # 1.5 s phrase at 120 bpm -> one 2 s bar
    looped = loop_notes(notes, bpm=120, duration=7)
    starts = [n.start for n in looped if n.pitch == 60]
    assert starts == [0.0, 2.0, 4.0, 6.0]
    assert all(n.start < 7 for n in looped)


def test_guide_chroma_has_no_phantom_fifth():
    import librosa
    y = render_guide([Note(60, 0, 2, 100)], sr=22050)   # a single C
    chroma = librosa.feature.chroma_stft(y=y, sr=22050).mean(axis=1)
    assert chroma.argmax() == 0                    # C
    assert chroma[7] < 0.3 * chroma[0]             # G (the fifth) stays weak


def test_keep_timing_leaves_notes_where_sung():
    raw = [Note(60, 1.03, 1.49, 90), Note(64, 1.52, 2.11, 90), Note(67, 2.17, 2.9, 90)]
    m = clean_melody(raw, bpm=120, keep_timing=True)
    assert [(n.start, n.end) for n in m.notes] == [(1.03, 1.49), (1.52, 2.11), (2.17, 2.9)]


def test_render_guide_length_and_range():
    y = render_guide(seq([60, 64, 67]), sr=16000)
    assert y.size > 16000 and abs(y).max() <= 0.81
