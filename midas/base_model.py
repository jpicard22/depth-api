import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """Base class for all models in MiDaS."""
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Must be implemented in subclass.")

    def load(self, path):
        """Load weights from a file."""
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict, strict=False)

    def save(self, path):
        """Save model weights to a file."""
        torch.save(self.state_dict(), path)
