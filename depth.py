import torch
import cv2
import numpy as np
import logging
import sys
import os

# Ajouter le dossier MiDaS au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'mida_repo'))

from midas.midas_net import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

logging.basicConfig(level=logging.INFO)

def generate_depth_map(input_path, output_path):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join("mida_repo", "weights", "midas_v21_small-70d6b9c8.pt")
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        model.to(device)
        model.eval()

        transform = Compose([
            Resize(
                384, 384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet()
        ])

        img = cv2.imread(input_path)
        if img is None:
            raise Exception(f"L'image {input_path} n'a pas pu être chargée.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_height, original_width = img.shape[:2]
        img_input = transform({"image": img})["image"]
        img_input = torch.from_numpy(img_input).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model.forward(img_input)
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

        logging.info(f"Carte de profondeur enregistrée à : {output_path}")

    except Exception as e:
        logging.error(f"Erreur lors du traitement : {e}")
        raise
