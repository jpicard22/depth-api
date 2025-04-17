import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3

from midas.blocks import FeatureFusionBlock_custom, Interpolate, _make_encoder


class MidasNet_small(nn.Module):
    def __init__(
        self,
        path,
        features=64,
        backbone="efficientnet_lite3",
        exportable=True,
        non_negative=True,
        blocks={'expand': True},
    ):
        super(MidasNet_small, self).__init__()

        use_pretrained = path is None

        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            use_pretrained=use_pretrained,
            exportable=exportable,
            hooks=[2, 3, 4, 5],
            blocks=blocks,
        )

        self.scratch.refinenet4 = FeatureFusionBlock_custom(features)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2),
            nn.Conv2d(features // 2, 1, kernel_size=1, stride=1, padding=0),
        )

        self.non_negative = non_negative

        if path is not None:
            self.load(path)

    def forward(self, x):
        layer_1, layer_2, layer_3, layer_4 = self.pretrained(x)

        path_4 = self.scratch.refinenet4(layer_4)
        path_3 = self.scratch.refinenet3(path_4, layer_3)
        path_2 = self.scratch.refinenet2(path_3, layer_2)
        path_1 = self.scratch.refinenet1(path_2, layer_1)

        out = self.scratch.output_conv(path_1)

        if self.non_negative:
            out = F.relu(out)

        return out

    def load(self, path):
        state_dict = torch.load(path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.load_state_dict(state_dict)
