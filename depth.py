import torch
import cv2
import numpy as np
from PIL import Image
import timm

def generate_depth_map(input_path, output_path):
    """
    Génère une carte de profondeur à partir d'une image en utilisant le modèle MiDaS.

    Args:
        input_path (str): Chemin vers l'image d'entrée.
        output_path (str): Chemin où enregistrer la carte de profondeur.
    """
    try:
        # Charger le modèle MiDaS
        model_type = "DPT_Large"
        midas = torch.hub.load("intel-isl/MiDaS", model_type)

        # Utiliser CUDA si disponible, sinon le CPU
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()

        # Charger le transformateur correspondant au modèle
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.small_transform if model_type == "DPT_Large" else midas_transforms.dpt_transform

        # Charger et prétraiter l'image
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        output_depth = cv2.normalize(depth_map, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8U)
        output_depth_color = cv2.applyColorMap(output_depth, cv2.COLORMAP_INFERNO)
        cv2.imwrite(output_path, output_depth_color)

    except Exception as e:
        print(f"Erreur lors de la génération de la carte de profondeur : {e}")
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