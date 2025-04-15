from flask import Flask, request, jsonify
import os
import urllib.request
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose

app = Flask(__name__)

MODEL_PATH = "weights/dpt_beit_large_384.pt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1UjDhAMJc0La1I_n7Kn_KRc8Y3hbRt4jn"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📦 Téléchargement du modèle depuis Google Drive...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Modèle téléchargé avec succès.")

download_model()

# Charger le modèle depuis le chemin local avec torch.hub
model_type = "DPT_BEiT_L_384"  # pour correspondre au fichier .pt
model = torch.hub.load("intel-isl/MiDaS", model_type, model_path=MODEL_PATH, trust_repo=True)
model.eval()


# Transformation d'image adaptée au modèle
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.dpt_transform

@app.route('/', methods=['POST'])
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
