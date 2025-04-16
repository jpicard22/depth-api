import torch
from midas.midas_net import MidasNet_small
from torchvision import models

def load_model(model_type: str, device: torch.device, model_path: str):
    if model_type == "midas_v21_small_256":
        # Charger ResNet50 localement
        resnet50 = models.resnet50(pretrained=False)  # Ne pas télécharger les poids par défaut
        # Charger les poids sur le CPU
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # Filtrer les clés pour s'assurer que nous ne chargeons que celles nécessaires
        model_state_dict = resnet50.state_dict()  # Obtenir l'état du modèle ResNet50
        state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict.keys()}  # Garder les clés correspondantes
        
        # Charger les poids filtrés dans ResNet50
        resnet50.load_state_dict(state_dict)
        resnet50.to(device)  # Transférer le modèle sur le périphérique approprié
        
        # Initialiser MidasNet_small avec ResNet50 comme backbone
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
