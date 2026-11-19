# Hum to Hollywood

Hum a melody, choose a style (film score, pop, rock...), and get back a full arranged track built around your tune.

```
Record hum -> Clean audio -> Hum to MIDI -> Melody clean-up -> Style prompt -> MusicGen Melody -> Post-process -> Play / download
```

## Project layout

| Path | Stage | What it does |
|---|---|---|
| `src/audio_in.py` | 1 | load / record, mono, resample, trim silence, noise reduction |
| `src/transcribe.py` | 2 | hum -> notes with Spotify Basic Pitch |
| `src/melody.py` | 3 | **the core**: stray-note removal, monophonic reduction, wobble merging, key detection (Krumhansl-Schmuckler) + snapping, tempo estimate, quantising, MIDI export, guide-track synthesis, piano roll |
| `src/prompts.py` | 4 | style presets -> text prompt, with detected tempo and key added |
| `src/generate.py` | 5 | MusicGen Melody via Hugging Face `transformers` |
| `src/postprocess.py` | 6 | loudness to -14 LUFS, fades, optional hum overlay, WAV/MP3 export |
| `src/pipeline.py` | all | runs every stage; also a CLI |
| `app/app.py` | 7 | Gradio web app |
| `notebooks/colab_generate.ipynb` | 5, 7 | run the real model on a free Colab GPU |
| `data/` | 8 | private training audio (git-ignored); `labels_template.csv` shows the label format |

## Setup (Windows, Python 3.10 or 3.11)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --no-deps basic-pitch   # skips TensorFlow; uses ONNX Runtime instead
pip install -r requirements.txt
```

`torch` and `transformers` (last lines of `requirements.txt`) are only needed for real generation. Leave them out on a laptop without a GPU.

## Run

```powershell
python app/app.py
```

Or from the command line:

```powershell
python -m src.pipeline my_hum.wav --style "Epic film score" --duration 15
```

Outputs land in `outputs/`: the track (WAV + MP3), the cleaned melody MIDI, the guide WAV, and a piano-roll PNG.

### Generation backend

`HTH_BACKEND` picks how stage 5 runs:

- `musicgen`: the real `facebook/musicgen-melody` model (about 6 GB download). Practical only on a GPU.
- `mock`: an instant placeholder (your cleaned melody looped over a simple beat) so you can build and test the app on a laptop. **Not AI-generated music.**

If unset, it uses `musicgen` when a CUDA GPU is present and `mock` otherwise. For real tracks without a local GPU, open `notebooks/colab_generate.ipynb` in Colab on a T4 GPU, or deploy to a Hugging Face Space with GPU hardware.

## Tests

```powershell
pytest -q
```

`tests/test_melody.py` checks each clean-up step on hand-built notes; `tests/test_pipeline.py` synthesises a vibrato "hum" and checks Basic Pitch plus clean-up recover the melody and key.

## Roadmap

- [x] Stages 1-3: hum to clean MIDI
- [x] Stages 4-7: prompts, generation, post-processing, Gradio app
- [ ] Evaluation on MIR-QBSH (note accuracy, pitch error) and melody faithfulness (hum vs output pitch contour)
- [ ] Deploy to Hugging Face Spaces, demo video
- [ ] Stage 8: fine-tune MusicGen on a labelled style set (see `data/labels_template.csv`)

## Licences

MusicGen weights are CC-BY-NC 4.0 (non-commercial; fine for a portfolio). Keep any copyrighted training audio in `data/`, which is git-ignored. Never commit or publish it.
