import sys
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Fonction de log avec encodage UTF-8 pour gérer les caractères spéciaux
def log(message):
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(message + "\n")

# Vérification des arguments
if len(sys.argv) != 3:
    log("❌ Mauvais nombre d'arguments")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

# Chargement du modèle MiDaS
log("📦 Chargement du modèle MiDaS avec trust_repo=True...")
try:
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True, trust_repo=True)
    model.eval()
except Exception as e:
    log(f"❌ Erreur lors du chargement du modèle MiDaS : {str(e)}")
    sys.exit(1)

# Transformation des images
transform = transforms.Compose([
    transforms.Resize(384),  # Redimensionner pour DPT_Large
    transforms.ToTensor(),
])

try:
    # Ouverture de l'image d'entrée
    img = Image.open(input_path).convert("RGB")  # Convertir en RGB au cas où

    # Appliquer la transformation
    img_input = transform(img).unsqueeze(0)

    # Calcul de la carte de profondeur
    with torch.no_grad():
        depth_map = model(img_input)

    # Sauvegarde en niveaux de gris
    depth_map = depth_map.squeeze().cpu().numpy()
    plt.imsave(output_path, depth_map, cmap='gray')

    log(f"✅ Carte de profondeur générée avec succès : {output_path}")

except Exception as e:
    log(f"❌ Erreur lors du traitement de l'image : {str(e)}")
    sys.exit(1)
