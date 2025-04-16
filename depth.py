import torch
import cv2
import numpy as np
from PIL import Image
import timm
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)

# Ajouter le chemin du dépôt MiDaS cloné au sys.path
sys.path.append(os.path.join(os.getcwd(), 'midas_repo'))

from midas.model_loader import load_model
from midas.transforms import MiDaS_small_transform  # Ou DPTTransform si vous utilisiez un autre modèle

def generate_depth_map(input_path, output_path):
    try:
        model_path = 'weights/midas_v21_small_256.pt'

        if not os.path.exists(model_path):
            logging.error(f"Fichier de poids non trouvé à : {model_path}")
            raise FileNotFoundError(f"Fichier de poids du modèle non trouvé à : {model_path}")
        else:
            logging.info(f"Chargement des poids depuis le fichier local : {model_path}")

        # Charger le modèle directement depuis le code local
        model_type = "MiDaS_small"
        midas = load_model(model_type, pretrained=False)
        midas.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()
        logging.info(f"Modèle {model_type} chargé depuis le code local.")

        # Charger le transformateur correspondant
        transform = MiDaS_small_transform(model_type) # Ou DPTTransform()
        logging.info("Transformateur MiDaS chargé.")

        # Charger et prétraiter l'image
        img = cv2.imread(input_path)
        logging.info(f"Image chargée depuis : {input_path}, shape: {img.shape}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_height, original_width = img.shape[:2]

        max_size = 512
        if max(original_height, original_width) > max_size:
            if original_height > original_width:
                new_height = max_size
                new_width = int(original_width * (new_height / original_height))
            else:
                new_width = max_size
                new_height = int(original_height * (new_width / new_width))
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            logging.info(f"Image redimensionnée de ({original_width}, {original_height}) à ({new_width}, {new_height})")

        input_batch = transform.apply_image(img).to(device) # Utiliser apply_image

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(original_height, original_width),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            logging.info("Carte de profondeur générée et redimensionnée.")

        depth_map = prediction.cpu().numpy()
        output_depth = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        output_depth_color = cv2.applyColorMap(output_depth, cv2.COLORMAP_INFERNO)
        cv2.imwrite(output_path, output_depth_color)
        logging.info(f"Carte de profondeur enregistrée à : {output_path}")

    except Exception as e:
        logging.error(f"Erreur lors du traitement : {e}")
        raise

if __name__ == '__main__':
    input_image_path = 'uploads/test.jpg'
    output_depth_path = 'processed/test_depth.png'
    try:
        generate_depth_map(input_image_path, output_depth_path)
        print(f"Carte de profondeur générée et enregistrée à : {output_depth_path}")
    except Exception as e:
        print(f"Erreur : {e}")