import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import os
import torch.nn.functional as F
from unet_part import *

class conv1_layers(nn.Module):
    def __init__(self, image_channels=1):
        super(conv1_layers, self).__init__()
        f_layers = []

        f_layers.append(nn.Conv2d(in_channels=image_channels, out_channels=96, kernel_size=3, stride=1, padding=1))
        f_layers.append(nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1))
        f_layers.append(nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1))
        f_layers.append(nn.Conv2d(in_channels=96, out_channels=48, kernel_size=1, stride=1, padding=0))
        f_layers.append(nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1))
        f_layers.append(nn.Conv2d(in_channels=48, out_channels=image_channels, kernel_size=3, stride=1, padding=1))

        self.feat = nn.Sequential(*f_layers)

    def forward(self, x):
        x = self.feat(x)
        return x


class conv2_layers(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(conv2_layers, self).__init__()
        layers = []

        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0))
        layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

class dynamic_filter(nn.Module):
    def __init__(self, kernel_size=9, image_channels=1):
        super(dynamic_filter, self).__init__()
        layers = []
        #encoder
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))

        #mid
        layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))

        self.encoder = nn.Sequential(*layers)

        layers2 = []
        layers2.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False))
        layers2.append(nn.LeakyReLU(inplace=True))
        layers2.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False))
        layers2.append(nn.LeakyReLU(inplace=True))
        layers2.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False))
        layers2.append(nn.LeakyReLU(inplace=True))

        self.decoder = nn.Sequential(*layers2)

        self.conv1 = nn.Conv2d(in_channels=128, out_channels=kernel_size*kernel_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        x_ = self.encoder(x)
        x_ = nn.Upsample(size=(x.size(2),x.size(3)))(x_)
        x_ = self.decoder(x_)
        x_ = self.conv1(x_)
        filter = self.softmax(x_)
        return filter

class TPN(nn.Module):
    def __init__(self, image_channels=1):
        super().__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(image_channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(image_channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(image_channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(image_channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.conv2_4 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(True)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(True)
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(True)
        )
        self.conv3_4 = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        N, C, h, w = x.size()

        size1 = x
        size2 = F.interpolate(x, size=(h // 2, w // 2))
        size4 = F.interpolate(x, size=(h // 4, w // 4))
        size8 = F.interpolate(x, size=(h // 8, w // 8))

        size1 = self.conv1_1(size1)
        size2 = self.conv1_2(size2)
        size4 = self.conv1_3(size4)
        size8 = self.conv1_4(size8)

        size1 = self.conv2_1(size1)
        size2 = self.conv2_2(size2)
        size4 = self.conv2_3(size4)
        size8 = self.conv2_4(size8)

        size1 = self.conv3_1(size1)
        size2 = self.conv3_2(size2)
        size4 = self.conv3_3(size4)
        size8 = self.conv3_4(size8)

        size2 = F.interpolate(size2, size=(h, w))
        size4 = F.interpolate(size4, size=(h, w))
        size8 = F.interpolate(size8, size=(h, w))

        concat = torch.cat((size1, size2, size4, size8), 1)

        return self.conv4(concat)
