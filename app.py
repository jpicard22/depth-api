from flask import Flask, request, jsonify
import os
import urllib.request
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch import nn

app = Flask(__name__)

MODEL_PATH = "weights/dpt_beit_large_384.pt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1UjDhAMJc0La1I_n7Kn_KRc8Y3hbRt4jn"

# Fonction pour télécharger le modèle si nécessaire
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📦 Téléchargement du modèle depuis Google Drive...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Modèle téléchargé avec succès.")

# Charger le modèle directement depuis le fichier .pt
def load_model():
    # Charger le modèle
    model = torch.load(MODEL_PATH)
    model.eval()  # Mettre le modèle en mode évaluation
    return model

# Route principale pour générer la carte de profondeur
@app.route('/', methods=['POST'])
def depth_map():
    # Charger le modèle si ce n'est pas déjà fait
    if not hasattr(depth_map, "model"):
        download_model()
        depth_map.model = load_model()

        # Transformation d'image adaptée au modèle
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        depth_map.transform = midas_transforms.dpt_transform

    if 'image' not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    # Traitement de l'image reçue
    img = Image.open(request.files['image']).convert('RGB')
    input_tensor = depth_map.transform(img).unsqueeze(0)

    with torch.no_grad():
        prediction = depth_map.model(input_tensor)
        depth = prediction.squeeze().cpu().numpy()

    # Normalisation de la carte de profondeur
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth_img = (depth * 255).astype(np.uint8)

    _, img_encoded = cv2.imencode('.png', depth_img)
    return img_encoded.tobytes(), 200, {'Content-Type': 'image/png'}

if __name__ == '__main__':
    # Utilisation du port dynamique sur Railway
    port = int(os.environ.get("PORT", 5000))  # Railway définit le port dynamiquement
    app.run(host='0.0.0.0', port=port)
