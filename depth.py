import torch
import cv2
import numpy as np
from PIL import Image
import timm
import logging
import os

logging.basicConfig(level=logging.INFO)

def generate_depth_map(input_path, output_path):
    """
    Génère une carte de profondeur à partir d'une image en utilisant le modèle MiDaS,
    avec un redimensionnement de l'image d'entrée pour optimiser les performances.

    Args:
        input_path (str): Chemin vers l'image d'entrée.
        output_path (str): Chemin où enregistrer la carte de profondeur.
    """
    try:
        model_type = "MiDaS_small"
        repo = "intel-isl/MiDaS"
        model_name = "midas_v21_small_256.pt"
        cached_file = os.path.join(torch.hub.get_dir(), f"{repo.replace('/', '_')}_{model_name}")

        logging.info(f"Chemin du fichier de poids mis en cache : {cached_file}")

        if not os.path.exists(cached_file):
            logging.info(f"Téléchargement des poids du modèle {model_name} depuis {repo}")
            midas = torch.hub.load(repo, model_type, trust_repo=True)
        else:
            logging.info(f"Utilisation du fichier de poids mis en cache : {cached_file}")
            midas = torch.hub.load(repo, model_type, trust_repo=True, pretrained=False)
            midas.load_state_dict(torch.load(cached_file))

        # Utiliser CUDA si disponible, sinon le CPU
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()

        # Charger le transformateur correspondant au modèle
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = midas_transforms.small_transform

        # Charger et prétraiter l'image
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_height, original_width = img.shape[:2]

        # Redimensionner l'image si elle est trop grande
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
                size=(original_height, original_width),  # Redimensionner la profondeur à la taille originale
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        output_depth = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        output_depth_color = cv2.applyColorMap(output_depth, cv2.COLORMAP_INFERNO)
        cv2.imwrite(output_path, output_depth_color)
        logging.info(f"Carte de profondeur générée et enregistrée à : {output_path}")

    except Exception as e:
        logging.error(f"Erreur lors de la génération de la carte de profondeur : {e}")
        raise

if __name__ == '__main__':
    # Exemple d'utilisation (pour test local)
    # Créez un dossier 'uploads' et placez une image nommée 'test.jpg' dedans
    input_image_path = 'uploads/test.jpg'
    output_depth_path = 'processed/test_depth.png'
    try:
        generate_depth_map(input_image_path, output_depth_path)
        print(f"Carte de profondeur générée et enregistrée à : {output_depth_path}")
    except Exception as e:
        print(f"Erreur : {e}")