import sys
import torch
import urllib
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# Charger le modèle local MiDaS
def load_midas_model():
    model_path = "weights/openvino_midas_v21_small_256.bin"
    if not os.path.exists(model_path):
        print(f"❌ Modèle introuvable : {model_path}")
        sys.exit(1)

    model = torch.hub.load(
        'intel-isl/MiDaS',
        'DPT_Large',
        pretrained=False
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


# Préparation de l'image
def preprocess(image_path):
    transform = torch.hub.load(
        'intel-isl/MiDaS',
        'transforms',
        pretrained=False
    ).dpt_transform

    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

# Main
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("❌ Usage : python depth.py <input_path> <output_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        model = load_midas_model()
        input_batch = preprocess(input_path)

        with torch.no_grad():
            prediction = model(input_batch)
            depth_map = prediction.squeeze().cpu().numpy()

        plt.imsave(output_path, depth_map, cmap='gray')
        print("✅ Carte de profondeur enregistrée :", output_path)

    except Exception as e:
        print("❌ Erreur :", str(e))
        sys.exit(1)
