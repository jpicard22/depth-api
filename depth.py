import torch
import cv2
import numpy as np
import logging
import sys
import os

# Ajouter le dossier parent au path pour que 'midas' soit trouvé comme un package de niveau supérieur
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from midas.midas_net import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

logging.basicConfig(level=logging.INFO)

def generate_depth_map(input_path, output_path):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ modèle dans weights/ à la racine
        model_weights = os.path.join("weights", "midas_v21_small_256.pt")

        model = MidasNet_small(
            model_weights,
            features=64,
            backbone="efficientnet_lite3",
            exportable=True,
            non_negative=True,
            blocks={'expand': True}
        )
        model.to(device)
        model.eval()

        transform = Compose([
            Resize(256, 256, resize_target=None, keep_aspect_ratio=True,
                   ensure_multiple_of=32, resize_method="upper_bound",
                   image_interpolation_method=cv2.INTER_CUBIC),
            NormalizeImage(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            PrepareForNet()
        ])

        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Image {input_path} non lisible.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_height, original_width = img.shape[:2]

        img_input = transform({"image": img})["image"]
        img_input = torch.from_numpy(img_input).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(img_input)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(original_height, original_width),
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        depth_normalized = cv2.normalize(depth_map, None, 255, 0,
                                         cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        cv2.imwrite(output_path, depth_colored)

        logging.info(f"Carte de profondeur enregistrée : {output_path}")

    except Exception as e:
        logging.error(f"Erreur : {e}")
        raise