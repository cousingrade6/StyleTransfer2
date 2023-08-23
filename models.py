import torchvision.models as models
import torch.nn as nn
from utils import *


class myVGG(nn.Module):
    def __init__(self, features, name_list):
        super(myVGG, self).__init__()
        self.features = features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        outs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)
        return outs


def ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, upsample=None,
              instance_norm=True, relu=True):
    layers = []
    if upsample:
        layers.append(nn.UpsamplingNearest2d(scale_factor=upsample))
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            stride=stride, padding=kernel_size // 2))
    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            ConvLayer(channels, channels, kernel_size=3, stride=1),
            ConvLayer(channels, channels, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        y = self.conv(x)
        y += x
        return y


class TransformNet(nn.Module):
    """风格转换网络"""

    def __init__(self, base=32):
        super(TransformNet, self).__init__()
        self.downsampling = nn.Sequential(
            ConvLayer(in_channels=3, out_channels=base, kernel_size=9),
            ConvLayer(base, 2 * base, kernel_size=3, stride=2),
            ConvLayer(2 * base, 4 * base, kernel_size=3, stride=2),
        )
        self.residuals = nn.Sequential(*[ResidualBlock(4 * base) for i in range(5)])
        self.upsampling = nn.Sequential(
            ConvLayer(4 * base, 2 * base, kernel_size=3, upsample=2),
            ConvLayer(2 * base, base, kernel_size=3, upsample=2),
            ConvLayer(base, 3, kernel_size=9, instance_norm=False, relu=False)
        )

    def forward(self, x):
        x = self.downsampling(x)
        x = self.residuals(x)
        x = self.upsampling(x)
        return x





