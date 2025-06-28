import gradio as gr
import numpy as np
import random
from diffusers import DiffusionPipeline
import torch
import os

# Detect hardware
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model (Fast model + appropriate precision)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    variant="fp16" if device == "cuda" else None
)
pipe.to(device)
pipe.enable_attention_slicing()  # for low VRAM optimization
pipe.set_progress_bar_config(disable=True)

# Constants
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

# 🔥 Faster inference (turbo mode)
def infer(prompt, negative_prompt, seed, randomize_seed,
          width, height, guidance_scale, num_inference_steps,
          progress=gr.Progress(track_tqdm=False)):

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator
    ).images[0]

    return image, seed

# Example prompts
examples = [
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A futuristic Indian cyber city at night, neon lights, cinematic",
    "Superhero tiger with fire armor in forest"
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 640px;
}
"""

# Gradio UI
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("## 🚀 Unstar AI — Super Fast Image Generator")

        with gr.Row():
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt", scale=3)
            run_button = gr.Button("⚡ Generate", variant="primary")

        result = gr.Image(label="Generated Image", type="pil")

        with gr.Accordion("🎛️ Advanced Settings", open=False):
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Optional")
            seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
            randomize_seed = gr.Checkbox(label="🎲 Randomize Seed", value=True)

            width = gr.Slider(label="Width", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=1024)
            height = gr.Slider(label="Height", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=1024)

            guidance_scale = gr.Slider(label="Guidance Scale", minimum=0.0, maximum=10.0, step=0.5, value=0.0)
            num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=10, step=1, value=1)

        gr.Examples(examples=examples, inputs=[prompt])

        run_button.click(fn=infer, inputs=[
            prompt, negative_prompt, seed, randomize_seed,
            width, height, guidance_scale, num_inference_steps
        ], outputs=[result, seed])

if __name__ == "__main__":
    demo.queue(max_size=5).launch()
