import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import os
import torch.nn.functional as F
from utils import *

class Model(nn.Module):
    def __init__(self, model_dir=None, model_name=[], guidance=None, kernel_size=9):
        super(Model, self).__init__()
        self.DNet = torch.load(os.path.join(model_dir, model_name[0]))
        self.SNet = torch.load(os.path.join(model_dir, model_name[1]))
        #self.JFNet = JointNet(kernel_size=kernel_size)
        self.guide = guidance

    def forward(self, x):
        r = self.DNet(x)
        d = x-r

        if self.guide is 'noisy':
            s = self.SNet(r, x)
        elif self.guide is 'denoised':
            s = self.SNet(r, d)

        out = s+d
        return out

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

class SNet_dfver1(nn.Module):
    def __init__(self, kernel_size=3, image_channels=1):
        super(SNet_dfver1, self).__init__()
        layers = []
        self.conv1 = conv1_layers()
        self.conv2 = conv1_layers()
        self.filter = dynamic_filter(image_channels=2)

    def forward(self, x, g):
        x_ = self.conv1(x)
        g_ = self.conv2(g)
        x_ = torch.cat((x_, g_), 1)
        filter = self.filter(x_)

        patches = F.pad(x, (4,4,4,4), "constant", 0)
        patches = patches.unfold(2,9,1).unfold(3,9,1)
        patches = patches.permute(0,1,4,5,2,3).squeeze(1).reshape(-1,9*9,x.size(2),x.size(3))
        out = torch.sum(patches*filter, 1, keepdim=True)
        return out

class SNet_dfver2(nn.Module):
    def __init__(self, kernel_size=3, image_channels=1):
        super(SNet_dfver2, self).__init__()
        layers = []
        self.conv1 = conv2_layers()
        self.filter_x = dynamic_filter(image_channels=1)
        self.filter_g = dynamic_filter(image_channels=1)

    def forward(self, x, g):
        filter_x = self.filter_x(x)
        filter_g = self.filter_g(g)

        patches = F.pad(x, (4,4,4,4), "constant", 0)
        patches = patches.unfold(2,9,1).unfold(3,9,1)
        patches = patches.permute(0,1,4,5,2,3).squeeze(1).reshape(-1,9*9,x.size(2),x.size(3))
        x_ = torch.sum(patches*filter_x, 1, keepdim=True)

        patches = F.pad(g, (4,4,4,4), "constant", 0)
        patches = patches.unfold(2,9,1).unfold(3,9,1)
        patches = patches.permute(0,1,4,5,2,3).squeeze(1).reshape(-1,9*9,x.size(2),x.size(3))
        g_ = torch.sum(patches*filter_g, 1, keepdim=True)

        out = self.conv1(torch.cat((x_,g_),1))
        return out

class TPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
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
