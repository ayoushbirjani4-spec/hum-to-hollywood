"""Stage 7: Gradio web app.

    python app/app.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from src import audio_in, generate, pipeline, prompts

MOCK_NOTE = ("⚠️ **Placeholder mode (no GPU):** the track is your cleaned melody over a fixed beat, "
             "so every style sounds the same. The melody, key, tempo and prompt are real. "
             "For real AI music in your chosen style, run on a GPU (see the Colab notebook in the README).")


def on_generate(audio, style, extra, duration, bpm, hum_level, seed):
    if audio is None:
        raise gr.Error("Record or upload a hum first.")
    y = audio_in.from_gradio(audio)
    try:
        r = pipeline.run(y, style=style, extra=extra or "", duration=duration,
                         bpm=bpm or None, hum_level=hum_level, seed=int(seed) if seed else None)
    except ValueError as e:
        raise gr.Error(str(e))
    info = (f"**{r.key} · {r.bpm:.0f} bpm** · {r.n_raw_notes} raw notes cleaned to {r.n_notes}\n\n"
            f"Prompt: *{r.prompt}*\n\n"
            f"Backend: `{r.backend}`")
    if r.backend == "mock":
        info = (MOCK_NOTE + "\n\n" + info)
    downloads = [r.files[k] for k in ("track_wav", "track_mp3", "midi") if k in r.files]
    return r.files["track_wav"], r.files["piano_roll"], r.files["guide_wav"], info, downloads


with gr.Blocks(title="Hum to Hollywood") as demo:
    gr.Markdown("# Hum to Hollywood\nHum a melody, pick a style, get a full arranged track built around your tune.")
    if generate.backend() == "mock":
        gr.Markdown(MOCK_NOTE)
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Your hum (5-15 seconds works best)")
            style = gr.Dropdown(list(prompts.PRESETS), value="Epic film score", label="Style", allow_custom_value=True)
            extra = gr.Textbox(label="Extra description (optional)", placeholder="e.g. with a female choir, melancholic")
            with gr.Accordion("Settings", open=False):
                duration = gr.Slider(5, 30, value=15, step=1, label="Length (seconds)")
                bpm = gr.Number(value=0, label="Tempo override (0 = detect)")
                hum_level = gr.Slider(0, 1, value=0, step=0.05, label="Mix your hum on top")
                seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
            btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            track = gr.Audio(label="Your track", type="filepath")
            info = gr.Markdown()
            roll = gr.Image(label="Detected melody", type="filepath")
            guide = gr.Audio(label="Cleaned melody (guide)", type="filepath")
            files = gr.File(label="Downloads", file_count="multiple")
    btn.click(on_generate, [audio, style, extra, duration, bpm, hum_level, seed], [track, roll, guide, info, files])
    gr.Markdown(f"Generation backend: `{generate.backend()}`. MusicGen weights are CC-BY-NC: non-commercial use only.")


if __name__ == "__main__":
    demo.launch()
