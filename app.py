import os
import torch
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, send_file
from torchvision.transforms import Compose

app = Flask(__name__)

# Chargement du modèle MiDaS
def load_model():
    model_type = "DPT_BEiT_L_384"
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    return model, transform

midas, transform = load_model()

@app.route("/", methods=["POST"])
def predict():
    if "image" not in request.files:
        return {"error": "Aucune image fournie"}, 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.size[::-1],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    output = prediction.cpu().numpy()
    output = (output - output.min()) / (output.max() - output.min())
    output = (output * 255).astype(np.uint8)

    output_path = "/tmp/depth.png"
    cv2.imwrite(output_path, output)

    return send_file(output_path, mimetype="image/png")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"✅ Démarrage de l'API Flask sur le port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
