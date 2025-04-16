import torch
from midas.midas_net import MidasNet_small
from torchvision import models

def load_model(model_type: str, device: torch.device, model_path: str):
    if model_type == "midas_v21_small_256":
        # Charger ResNet50 localement
        resnet50 = models.resnet50(pretrained=False)  # Ne pas télécharger les poids par défaut
        resnet50.load_state_dict(torch.load(model_path))  # Charger les poids depuis le fichier local
        resnet50.to(device)
        
        # Maintenant, initialiser MidasNet_small avec ResNet50 comme backbone
        model = MidasNet_small(
            path=model_path,
            features=64,
            backbone=resnet50,  # Utiliser le ResNet50 chargé localement
            non_negative=True,
            blocks={'expand': True}
        )
    else:
        raise ValueError(f"Model type {model_type} not supported.")

    model.to(device)
    model.eval()
    return model
