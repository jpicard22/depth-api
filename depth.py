import torch
import cv2
import numpy as np
from PIL import Image
import timm
import logging
import os

logging.basicConfig(level=logging.INFO)

def generate_depth_map(input_path, output_path):
    try:
        model_path = 'weights/midas_v21_small_256.pt'
        if not os.path.exists(model_path):
            logging.error(f"Fichier de poids non trouvé à : {model_path}")
            raise FileNotFoundError(f"Fichier de poids non trouvé à : {model_path}")
        else:
            logging.info(f"Chargement du modèle depuis : {model_path}")
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True, pretrained=False)
        midas.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()
        logging.info("Modèle MiDaS chargé.")

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = midas_transforms.small_transform
        logging.info("Transformateur MiDaS chargé.")

        img = cv2.imread(input_path)
        logging.info(f"Image chargée depuis : {input_path}, shape: {img.shape}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_height, original_width = img.shape[:2]

        max_size = 256  # Réduire encore la taille pour tester
        if max(original_height, original_width) > max_size:
            # ... (votre logique de redimensionnement) ...
            logging.info(f"Image redimensionnée à ({img.shape[1]}, {img.shape[0]})")

        input_batch = transform(img).to(device)
        logging.info("Image transformée et déplacée vers le dispositif.")

        with torch.no_grad():
            prediction = midas(input_batch)
            logging.info("Prédiction MiDaS effectuée.")
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(original_height, original_width),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            logging.info("Carte de profondeur redimensionnée à la taille originale.")

        depth_map = prediction.cpu().numpy()
        output_depth = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        output_depth_color = cv2.applyColorMap(output_depth, cv2.COLORMAP_INFERNO)
        cv2.imwrite(output_path, output_depth_color)
        logging.info(f"Carte de profondeur enregistrée à : {output_path}")

    except Exception as e:
        logging.error(f"Erreur lors du traitement : {e}")
        raise

if __name__ == '__main__':
    # Exemple d'utilisation (pour test local)
    input_image_path = 'uploads/test.jpg'
    output_depth_path = 'processed/test_depth.png'
    try:
        generate_depth_map(input_image_path, output_depth_path)
        print(f"Carte de profondeur générée et enregistrée à : {output_depth_path}")
    except Exception as e:
        print(f"Erreur : {e}")