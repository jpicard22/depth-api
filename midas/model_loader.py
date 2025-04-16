# midas/model_loader.py

import torch
from midas.midas_net import MidasNet_small  # Importer MidasNet_small

def load_model(model_type: str, device: torch.device, model_path: str):
    if model_type == "midas_v21_small_256":
        model = MidasNet_small(
            path=model_path,
            features=64,
            backbone="efficientnet_lite3",  # Adapter selon ton modèle
            exportable=False,
            non_negative=True,
            blocks={'expand': True}
        )
    else:
        raise ValueError(f"Model type {model_type} not supported.")

    model.to(device)
    model.eval()
    return model
