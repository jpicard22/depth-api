import os
import sys
import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
from io import BytesIO

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
transform = midas_transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"] else midas_transforms.small_transform

print("✅ Modèle chargé avec succès", file=sys.stderr)

@app.route("/", methods=["GET"])
def ping():
    return jsonify({"message": "API MiDaS opérationnelle 🚀"})

@app.route("/", methods=["POST"])
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

    return send_file(BytesIO(buffer.tobytes()), mimetype="image/png")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"✅ Lancement sur le port {port}", file=sys.stderr)
    app.run(host="0.0.0.0", port=port)
