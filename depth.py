import torch
import cv2
import os
import urllib.request
from torchvision.transforms import Compose
from PIL import Image
import numpy as np

# 🔧 Chemins
MODEL_PATH = "weights/dpt_beit_large_384.pt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1UjDhAMJc0La1I_n7Kn_KRc8Y3hbRt4jn"
INPUT_PATH = "public/uploads/input.jpg"
OUTPUT_PATH = "public/processed/depth.png"

# 📥 Téléchargement du modèle si absent
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📦 Téléchargement du modèle depuis Google Drive...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Modèle téléchargé.")

download_model()

# 🔍 Charger modèle MiDaS avec modèle local
model_type = "DPT_BEiT_L_384"
model = torch.hub.load("intel-isl/MiDaS", model_type, model_path=MODEL_PATH, trust_repo=True)
model.eval()

# 🔧 Chargement des transforms MiDaS
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.dpt_transform

# 📸 Lecture de l'image
img = Image.open(INPUT_PATH).convert("RGB")
input_tensor = transform(img).unsqueeze(0)

# 🔍 Prédiction de la profondeur
with torch.no_grad():
    prediction = model(input_tensor)
    depth = prediction.squeeze().cpu().numpy()

# 🎨 Normalisation
depth = (depth - depth.min()) / (depth.max() - depth.min())
depth_img = (depth * 255).astype(np.uint8)

# 💾 Sauvegarde de l’image
cv2.imwrite(OUTPUT_PATH, depth_img)
print("✅ Carte de profondeur générée :", OUTPUT_PATH)
