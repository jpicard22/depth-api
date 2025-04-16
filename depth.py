import torch
import cv2
import numpy as np
import logging
import os
from midas.model_loader import load_model

logging.basicConfig(level=logging.INFO)

def generate_depth_map(input_path, output_path):
    try:
        # Définir le chemin vers le fichier de poids dans le dossier 'weights'
        model_path = 'weights/midas_v21_small_256.pt'
        model_type = 'midas_v21_small_256'

        # Vérifier si le fichier de poids existe
        if not os.path.exists(model_path):
            logging.error(f"Fichier de poids du modèle non trouvé à : {model_path}")
            raise FileNotFoundError(f"Fichier de poids du modèle non trouvé à : {model_path}")
        else:
            logging.info(f"Chargement du modèle depuis le fichier local : {model_path}")

        # Sélection du device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Charger le modèle depuis midas/model_loader.py
        midas = load_model(model_type, device, model_path)
        midas.eval()

        # Charger les transforms
        from midas.transforms import small_transform
        transform = small_transform

        # Charger et prétraiter l'image
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_height, original_width = img.shape[:2]

        input_tensor = transform({"image": img})["image"].to(device).unsqueeze(0)

        # Prédiction
        with torch.no_grad():
            prediction = midas(input_tensor)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(original_height, original_width),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Normalisation et sauvegarde
        depth_map = prediction.cpu().numpy()
        output_depth = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        output_depth_color = cv2.applyColorMap(output_depth, cv2.COLORMAP_INFERNO)
        cv2.imwrite(output_path, output_depth_color)
        logging.info(f"Carte de profondeur générée et enregistrée à : {output_path}")

    except Exception as e:
        logging.error(f"Erreur lors du traitement : {e}")
        raise

if __name__ == '__main__':
    input_image_path = 'uploads/test.jpg'
    output_depth_path = 'processed/test_depth.png'
    try:
        generate_depth_map(input_image_path, output_depth_path)
        print(f"Carte de profondeur générée à : {output_depth_path}")
    except Exception as e:
        print(f"Erreur : {e}")
