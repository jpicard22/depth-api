import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def generate_depth_map(input_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from torch.hub
    model_type = "DPT_Large"  # You can use DPT_Hybrid or MiDaS_small as alternatives
    model = torch.hub.load("intel-isl/MiDaS", model_type).to(device)
    model.eval()

    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform if "DPT" in model_type else midas_transforms.small_transform

    # Load image
    img = cv2.imread(input_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Transform image
    input_tensor = transform(img_rgb).to(device).unsqueeze(0)

    # Predict depth
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize and save
    depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_uint8 = depth_map_norm.astype(np.uint8)
    cv2.imwrite(output_path, depth_map_uint8)
