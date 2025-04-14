import os
import sys
import torch
import cv2
import urllib.request
import numpy as np
from flask import Flask, request, jsonify
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image

# Initialisation Flask
app = Flask(__name__)

# 🔄 Préchargement du modèle MiDaS
print("⏳ Chargement du modèle MiDaS...", file=sys.stderr)
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

print("✅ Modèle chargé avec succès", file=sys.stderr)

# 🔁 Route principale de test
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "API MiDaS opérationnelle 🚀"})

# 🔄 Route pour traitement d'image
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")

    input_image = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_image.unsqueeze(0))
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.size[::-1],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_uint8 = depth_map_normalized.astype(np.uint8)
    _, buffer = cv2.imencode(".png", depth_map_uint8)

    return buffer.tobytes(), 200, {"Content-Type": "image/png"}

# 🚀 Lancement
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"✅ PORT Railway = {port}", file=sys.stderr)
    app.run(host="0.0.0.0", port=port)
