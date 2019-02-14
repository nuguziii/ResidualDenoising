import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

class DNet(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DNet, self).__init__()
        kernel_size = 3
        padding = 1
        d_layers = []

        d_layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        d_layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            d_layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            d_layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            d_layers.append(nn.ReLU(inplace=True))
        d_layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*d_layers)

        self._initialize_weights()

    def forward(self, x):
        out = self.dncnn(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class SNet(nn.Module):
    def __init__(self, image_channels=1):
        super(SNet, self).__init__()
        kernel_size = 3
        padding = 1

        Struct_layers = []

        Struct_layers.append(nn.Conv2d(in_channels=image_channels, out_channels=96, kernel_size=9, stride=1, padding=2))
        Struct_layers.append(nn.Conv2d(in_channels=96, out_channels=48, kernel_size=1, stride=1, padding=2))
        Struct_layers.append(nn.Conv2d(in_channels=48, out_channels=image_channels, kernel_size=5, stride=1, padding=2))

        self.SNet = nn.Sequential(*Struct_layers)

        self._initialize_weights()

    def forward(self, x, g):
        x = self.SNet(x)
        x = x.add(g)
        out = self.SNet(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class JFNet(nn.Module):
    def __init__(self, image_channels=1):
        super(JFNet, self).__init__()
        kernel_size = 3
        padding = 1

        f_layers = []
        j_layers = []

        f_layers.append(nn.Conv2d(in_channels=image_channels, out_channels=96, kernel_size=9, stride=1, padding=2))
        f_layers.append(nn.Conv2d(in_channels=96, out_channels=48, kernel_size=1, stride=1, padding=2))
        f_layers.append(nn.Conv2d(in_channels=48, out_channels=image_channels, kernel_size=5, stride=1, padding=2))

        j_layers.append(nn.Conv2d(in_channels=image_channels*2, out_channels=64, kernel_size=9, stride=1, padding=0))
        j_layers.append(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0))
        j_layers.append(nn.Conv2d(in_channels=32, out_channels=image_channels, kernel_size=5, stride=1, padding=0))

        self.feat = nn.Sequential(*f_layers)
        self.joint = nn.Sequential(*j_layers)

        self._initialize_weights()

    def forward(self, x, d, g):
        x = self.feat(x.add(g))
        d = self.feat(d)
        x = torch.cat((x, d), 1)
        out = self.joint(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
