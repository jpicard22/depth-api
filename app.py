from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose

app = Flask(__name__)

# Charger directement le modèle supporté par torch.hub
model_type = "DPT_Large"  # ce modèle est dispo via torch.hub
model = torch.hub.load("intel-isl/MiDaS", model_type)
model.eval()

# Récupérer les bonnes transformations
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

@app.route("/", methods=["POST"])
def depth_map():
    if 'image' not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    img = Image.open(request.files['image']).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_tensor)
        depth = prediction.squeeze().cpu().numpy()

    # Normalisation de la carte de profondeur
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth_img = (depth * 255).astype(np.uint8)

    _, img_encoded = cv2.imencode('.png', depth_img)
    return img_encoded.tobytes(), 200, {'Content-Type': 'image/png'}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
