from flask import Flask, request, jsonify, send_file
import torch
from diffusers import StableDiffusionPipeline
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import uuid
import os

app = Flask(__name__)
os.makedirs("outputs", exist_ok=True)

image_pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

video_pipe = pipeline(Tasks.text_to_video_synthesis, model="damo/text-to-video-synthesis", device="cuda" if torch.cuda.is_available() else "cpu")

@app.route("/generate-image", methods=["POST"])
def generate_image():
    prompt = request.json.get("prompt")
    image = image_pipe(prompt).images[0]
    filename = f"outputs/{uuid.uuid4()}.png"
    image.save(filename)
    return send_file(filename, mimetype='image/png')

@app.route("/generate-video", methods=["POST"])
def generate_video():
    prompt = request.json.get("prompt")
    output_path = video_pipe({'text': prompt})['output_video']
    return send_file(output_path, mimetype='video/mp4')

@app.route("/", methods=["GET"])
def home():
    return "Unstar AI Backend Running Successfully"

if __name__ == "__main__":
    app.run(debug=False)
