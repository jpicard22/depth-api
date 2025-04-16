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
        # Définir le chemin vers le fichier de poids dans le dossier 'weights'
        model_path = 'weights/midas_v21_small_256.pt'

        # Vérifier si le fichier de poids existe
        if not os.path.exists(model_path):
            logging.error(f"Fichier de poids du modèle non trouvé à : {model_path}")
            raise FileNotFoundError(f"Fichier de poids du modèle non trouvé à : {model_path}")
        else:
            logging.info(f"Chargement du modèle depuis le fichier local : {model_path}")

        # Charger le modèle directement depuis le fichier de poids
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True, pretrained=False)
        midas.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()

        # Charger le transformateur
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = midas_transforms.small_transform

        # Charger et prétraiter l'image
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_height, original_width = img.shape[:2]

        # Redimensionner l'image si elle est trop grande (facultatif, mais recommandé)
        max_size = 512
        if max(original_height, original_width) > max_size:
            if original_height > original_width:
                new_height = max_size
                new_width = int(original_width * (new_height / original_height))
            else:
                new_width = max_size
                new_height = int(original_height * (new_width / original_width))
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            logging.info(f"Image redimensionnée de ({original_width}, {original_height}) à ({new_width}, {new_height})")

        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(original_height, original_width),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        output_depth = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        output_depth_color = cv2.applyColorMap(output_depth, cv2.COLORMAP_INFERNO)
        cv2.imwrite(output_path, output_depth_color)
        logging.info(f"Carte de profondeur générée et enregistrée à : {output_path}")

    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle ou de la génération de la carte de profondeur : {e}")
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