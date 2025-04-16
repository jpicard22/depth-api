import torch
import urllib.request
import cv2
import os
import numpy as np
from torchvision.transforms import Compose
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet

def generate_depth_map(input_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Télécharger le modèle s’il n’est pas déjà là
    model_type = "DPT_Large"  # ou "DPT_Hybrid" selon préférences
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval()
    model.to(device)

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform if model_type.startswith("DPT") else midas_transforms.small_transform

    # Lire l’image
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalisation pour affichage
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_vis = 255 * (depth_map - depth_min) / (depth_max - depth_min)
    depth_vis = depth_vis.astype("uint8")

    # Sauvegarder l’image de profondeur
    cv2.imwrite(output_path, depth_vis)
