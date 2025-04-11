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
log("Chargement du modèle MiDaS...")
model =  torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", pretrained=True)
model.eval()

# Transformation des images
transform = transforms.Compose([
    transforms.Resize(384),  # On redimensionne à 384 pour améliorer la qualité
    transforms.ToTensor(),  # Convertir en tensor
])

# transform = transforms.Compose([
#     transforms.Resize(384),  # Redimensionner l'image (384x384 pour ce modèle)
#     transforms.ToTensor(),   # Convertir en tensor
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalisation pour le modèle
# ])


try:
    # Ouverture de l'image d'entrée
    img = Image.open(input_path)

    # Appliquer la transformation (normalisation à 255)
    img_input = transform(img).unsqueeze(0)  # Ajouter une dimension pour le batch

    # Calcul de la carte de profondeur
    with torch.no_grad():
        depth_map = model(img_input)

    # Sauvegarder la carte de profondeur en niveaux de gris
    depth_map = depth_map.squeeze().cpu().numpy()
    plt.imsave(output_path, depth_map, cmap='gray')  # Utilisation de 'gray' pour les niveaux de gris
    # depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


    log(f"✅ Carte de profondeur générée avec succès : {output_path}")

except Exception as e:
    log(f"❌ Erreur lors du traitement de l'image : {str(e)}")
    sys.exit(1)
