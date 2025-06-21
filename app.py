from flask import Flask, request, send_file
import torch
from diffusers import StableDiffusionPipeline
import uuid
import os

app = Flask(__name__)
os.makedirs("outputs", exist_ok=True)

# ✅ Load lightweight pipeline for CPU
image_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    use_auth_token=True  # Optional if using private model
).to("cpu")

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
