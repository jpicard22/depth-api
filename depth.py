import sys
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

def log(message):
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(message + "\n")

if len(sys.argv) != 3:
    log("❌ Mauvais nombre d'arguments")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

log("Chargement du modèle MiDaS...")

model =  torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", pretrained=True)
# model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(384),
    transforms.ToTensor(),
])

try:
    img = Image.open(input_path)
    img = img.resize((640, 480))  # Redimensionner l'image pour le traitement

    with torch.no_grad():
        depth_map = model(img_input)

    depth_map = depth_map.squeeze().cpu().numpy()
    plt.imsave(output_path, depth_map, cmap='gray')

    log(f"✅ Carte de profondeur générée avec succès : {output_path}")

except Exception as e:
    log(f"❌ Erreur lors du traitement de l'image : {str(e)}")
    sys.exit(1)
