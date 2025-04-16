import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import logging
import os
import urllib.request
import types

logging.basicConfig(level=logging.INFO)

def generate_depth_map(input_path, output_path):
    try:
        model_path = 'weights/midas_v21_small_256.pt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fichier de poids du modèle non trouvé à : {model_path}")
        else:
            logging.info(f"Chargement du modèle depuis le fichier local : {model_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Charger le modèle (MiDaS Small 256)
        model = torch.load(model_path, map_location=device)
        if isinstance(model, dict) and 'state_dict' in model:
            state_dict = model['state_dict']
        else:
            state_dict = model

        from midas.model_loader import load_model  # Nécessite d’avoir le script `midas/model_loader.py`
        model = load_model("midas_v21_small_256", device, model_path)
        model.eval()

        # Définir les transformations (équivalent du small_transform de MiDaS)
        transform = Compose([
            Resize(256, interpolation=cv2.INTER_CUBIC),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        ])

        # Charger et prétraiter l'image
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_height, original_width = img.shape[:2]

        img_input = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(img_input)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(original_height, original_width),
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        output_depth = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        output_depth_color = cv2.applyColorMap(output_depth, cv2.COLORMAP_INFERNO)
        cv2.imwrite(output_path, output_depth_color)
        logging.info(f"Carte de profondeur générée et enregistrée à : {output_path}")

    except Exception as e:
        logging.error(f"Erreur : {e}")
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