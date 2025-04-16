import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ConvBlock(features, features)
        self.resConfUnit2 = ConvBlock(features, features)

    def forward(self, *xs):
        output = xs[0]
        for x in xs[1:]:
            output = output + x
        output = self.resConfUnit1(output)
        output = self.resConfUnit2(output)
        return output

def _make_encoder(backbone="resnet50", features=256, use_pretrained=True):
    model = models.resnet50(pretrained=use_pretrained)

    layer1 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1)
    layer2 = model.layer2
    layer3 = model.layer3
    layer4 = model.layer4

    scratch = nn.Module()
    scratch.layer1_rn = ConvBlock(256, features)
    scratch.layer2_rn = ConvBlock(512, features)
    scratch.layer3_rn = ConvBlock(1024, features)
    scratch.layer4_rn = ConvBlock(2048, features)

    class Pretrained(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = layer1
            self.layer2 = layer2
            self.layer3 = layer3
            self.layer4 = layer4

    return Pretrained(), scratch
