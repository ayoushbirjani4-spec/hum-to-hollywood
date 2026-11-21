"""Stage 7: Gradio web app.

    python app/app.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from src import audio_in, generate, pipeline, prompts

MOCK_NOTE = ("**Placeholder mode (no GPU):** the track is your cleaned melody over a fixed beat, "
             "so every style sounds the same. The melody, key, tempo and prompt are real. "
             "For real AI music in your chosen style, run on a GPU (see the Colab notebook in the README).")


def on_generate(audio, keep_voice, style, extra, duration, bpm, seed, guidance, model):
    if audio is None:
        raise gr.Error("Record or upload your singing or humming first.")
    y = audio_in.from_gradio(audio)
    try:
        r = pipeline.run(y, style=style, extra=extra or "", duration=duration,
                         bpm=bpm or None, seed=int(seed) if seed else None,
                         guidance_scale=guidance, model=model, keep_voice=keep_voice)
    except ValueError as e:
        raise gr.Error(str(e))
    m = r.tune_match
    tune = (f"**Tune match: {m.score:.0%}** (a track that ignored your tune would score about {m.baseline:.0%})\n\n"
            if m else "")
    info = (tune + f"**{r.key} · {r.bpm:.0f} bpm** · {r.n_raw_notes} raw notes cleaned to {r.n_notes}\n\n"
            f"Prompt: *{r.prompt}*\n\n"
            f"Backend: `{r.backend}`")
    if r.backend == "mock":
        info = (MOCK_NOTE + "\n\n" + info)
    downloads = [r.files[k] for k in ("track_wav", "track_mp3", "midi") if k in r.files]
    return r.files["track_wav"], r.files["piano_roll"], r.files["guide_wav"], info, downloads


with gr.Blocks(title="Hum to Hollywood") as demo:
    gr.Markdown("# Hum to Hollywood\nSing or hum your song, pick a style, and get it back as a full arranged track")
    if generate.backend() == "mock":
        gr.Markdown(MOCK_NOTE)
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["microphone", "upload"], type="numpy",
                             label="Your song (sing or hum, up to 30 seconds)")
            keep_voice = gr.Checkbox(value=True, label="Keep my voice: the music is built around your singing")
            style = gr.Dropdown(list(prompts.PRESETS), value="Epic film score", label="Style", allow_custom_value=True)
            extra = gr.Textbox(label="Extra description (optional)", placeholder="e.g. with a female choir, melancholic")
            with gr.Accordion("Settings", open=False):
                duration = gr.Slider(5, 30, value=15, step=1,
                                     label="Length in seconds (only without your voice; with it, the song lasts as long as your singing)")
                bpm = gr.Number(value=0, label="Tempo override (0 = detect)")
                seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                guidance = gr.Slider(1, 8, value=3, step=0.5,
                                     label="Follow strength (higher sticks closer to your tune and style; above ~6 can sound harsh)")
                model = gr.Radio(list(generate.MODELS), value=list(generate.MODELS)[0],
                                 label="Model (Large is a one-time ~10.5 GB download and slower)")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            track = gr.Audio(label="Your track", type="filepath")
            info = gr.Markdown()
            roll = gr.Image(label="Detected melody", type="filepath")
            guide = gr.Audio(label="Cleaned melody (guide)", type="filepath")
            files = gr.File(label="Downloads", file_count="multiple")
    btn.click(on_generate, [audio, keep_voice, style, extra, duration, bpm, seed, guidance, model],
              [track, roll, guide, info, files])
    gr.Markdown(f"Generation backend: `{generate.backend()}`. MusicGen weights are CC-BY-NC: non-commercial use only.")


if __name__ == "__main__":
    demo.launch()
