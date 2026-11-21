# Graph Report - Hum to Hollywood  (2026-11-21)

## Corpus Check
- 14 files · ~4,639 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 4 file(s) not represented in the graph (top: (none) 2, .csv 1, .ipynb 1)

## Summary
- 126 nodes · 270 edges · 12 communities (10 shown, 2 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 11 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `fc96bc28`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- pipeline.py
- transcribe.py
- numpy
- test_melody.py
- Note
- Hum to Hollywood
- generate.py
- detect_key
- melody.py
- clean_melody
- remove_strays

## God Nodes (most connected - your core abstractions)
1. `Note` - 24 edges
2. `run()` - 18 edges
3. `clean_melody()` - 15 edges
4. `merge_wobbles()` - 10 edges
5. `detect_key()` - 8 edges
6. `snap_to_key()` - 8 edges
7. `render_guide()` - 8 edges
8. `hum_to_melody()` - 8 edges
9. `generate()` - 7 edges
10. `Key` - 7 edges

## Surprising Connections (you probably didn't know these)
- `on_generate()` --calls--> `from_gradio()`  [EXTRACTED]
  app/app.py → src/audio_in.py
- `test_merge_keeps_real_steps()` --calls--> `merge_wobbles()`  [EXTRACTED]
  tests/test_melody.py → src/melody.py
- `test_detect_key_major_and_minor()` --calls--> `detect_key()`  [EXTRACTED]
  tests/test_melody.py → src/melody.py
- `test_clean_melody_end_to_end_on_notes()` --calls--> `clean_melody()`  [EXTRACTED]
  tests/test_melody.py → src/melody.py
- `seq()` --calls--> `Note`  [EXTRACTED]
  tests/test_melody.py → src/transcribe.py

## Import Cycles
- None detected.

## Communities (12 total, 2 thin omitted)

### Community 0 - "pipeline.py"
Cohesion: 0.11
Nodes (9): on_generate(), hum_to_melody(), main(), Result, run(), build_prompt(), fake_hum(), melody_result() (+1 more)

### Community 1 - "transcribe.py"
Cohesion: 0.14
Nodes (6): clean_audio(), from_gradio(), load_audio(), save_wav(), _get_model(), transcribe()

### Community 2 - "numpy"
Cohesion: 0.36
Nodes (5): export(), fade(), finish(), mix_hum(), normalize_loudness()

### Community 3 - "test_melody.py"
Cohesion: 0.29
Nodes (8): make_monophonic(), render_guide(), seq(), test_clean_melody_end_to_end_on_notes(), test_detect_key_major_and_minor(), test_merge_keeps_real_steps(), test_monophonic_keeps_stronger_overlap(), test_render_guide_length_and_range()

### Community 4 - "Note"
Cohesion: 0.27
Nodes (6): merge_wobbles(), quantize(), Note, test_merge_keeps_reattacked_repeat(), test_merge_wobbles_joins_split_held_note(), test_quantize_aligns_to_grid()

### Community 5 - "Hum to Hollywood"
Cohesion: 0.22
Nodes (8): Generation backend, Hum to Hollywood, Licences, Project layout, Roadmap, Run, Setup (Windows, Python 3.10 or 3.11), Tests

### Community 6 - "generate.py"
Cohesion: 0.33
Nodes (4): backend(), generate(), _load(), _mock()

### Community 7 - "detect_key"
Cohesion: 0.32
Nodes (4): detect_key(), Key, snap_to_key(), test_snap_to_key_moves_off_key_notes()

### Community 8 - "melody.py"
Cohesion: 0.38
Nodes (3): CleanMelody, piano_roll(), to_midi()

### Community 9 - "clean_melody"
Cohesion: 0.47
Nodes (3): clean_melody(), detect_onsets(), estimate_tempo()

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run()` connect `pipeline.py` to `transcribe.py`, `numpy`, `test_melody.py`, `generate.py`, `melody.py`?**
  _High betweenness centrality (0.144) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `Note` (e.g. with `CleanMelody` and `detect_key()`) actually correct?**
  _`Note` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Should `pipeline.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1103448275862069 - nodes in this community are weakly interconnected._
- **Why does `Note` connect `Note` to `transcribe.py`, `test_melody.py`, `detect_key`, `melody.py`, `clean_melody`, `remove_strays`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Should `transcribe.py` be split into smaller, more focused modules?**
  _Cohesion score 0.14285714285714285 - nodes in this community are weakly interconnected._
- **Why does `hum_to_melody()` connect `pipeline.py` to `transcribe.py`, `clean_melody`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._