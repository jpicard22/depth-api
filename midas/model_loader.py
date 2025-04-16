import torch
from midas.midas_net import MidasNet_small
from torchvision import models

def load_model(model_type: str, device: torch.device, model_path: str):
    if model_type == "midas_v21_small_256":
        # Charger ResNet50 avec des poids pré-entraînés d'ImageNet
        resnet50 = models.resnet50(weights='IMAGENET1K_V2')  # Utiliser les poids pré-entraînés
        resnet50.to(device)  # Transférer le modèle sur le périphérique approprié

        # Maintenant, initialiser MidasNet_small avec ResNet50 comme backbone
        model = MidasNet_small(
            path=model_path,
            features=64,
            backbone=resnet50,  # Utiliser le ResNet50 chargé avec des poids pré-entraînés
            non_negative=True,
            blocks={'expand': True}
        )
    else:
        raise ValueError(f"Model type {model_type} not supported.")

    model.to(device)
    model.eval()
    return model
