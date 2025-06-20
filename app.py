from flask import Flask, request, send_file
import torch
from diffusers import StableDiffusionPipeline
import uuid
import os

app = Flask(__name__)
os.makedirs("outputs", exist_ok=True)

# Load Stable Diffusion pipeline (CPU friendly version)
image_pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32  # ✅ use float32 for CPU
).to("cpu")  # ✅ force CPU for Render free tier

@app.route("/generate-image", methods=["POST"])
def generate_image():
    prompt = request.json.get("prompt")
    image = image_pipe(prompt).images[0]
    filename = f"outputs/{uuid.uuid4()}.png"
    image.save(filename)
    return send_file(filename, mimetype='image/png')

@app.route("/", methods=["GET"])
def home():
    return "✅ Unstar AI Image Generator Backend Running"

# ✅ Dynamic port support for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
