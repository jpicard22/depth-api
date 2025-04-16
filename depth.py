import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose

# Chargement du modèle MiDaS au démarrage
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
midas.eval()

# Chargement des transformations nécessaires pour MiDaS
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.dpt_transform

def generate_depth_map(image_path, output_path):
    """
    Fonction pour générer une carte de profondeur à partir d'une image
    """
    # Lecture de l'image avec OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Conversion du BGR au RGB

    # Transformation de l'image pour le modèle MiDaS
    input_image = Image.fromarray(image)
    input_tensor = transform(input_image).unsqueeze(0)  # Ajouter une dimension batch

    # Exécution du modèle MiDaS pour générer la carte de profondeur
    with torch.no_grad():
        prediction = midas(input_tensor)  # Prédiction de la carte de profondeur
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),  # Ajouter une dimension de canal
            size=input_image.size[::-1],  # Taille de l'image d'origine (hauteur, largeur)
            mode="bicubic",
            align_corners=False,
        ).squeeze()  # Suppression de la dimension inutile

    # Normalisation des résultats
    output = prediction.cpu().numpy()  # Conversion vers numpy
    normalized = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)  # Normalisation des valeurs
    depth_map = np.uint8(normalized)  # Conversion en format uint8 pour l'image

    # Sauvegarde de la carte de profondeur
    cv2.imwrite(output_path, depth_map)
