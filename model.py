import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import os
import torch.nn.functional as F
from utils import *

class Model(nn.Module):
    def __init__(self, model_dir=None, model_name=[]):
        super(Model, self).__init__()
        self.DNet = torch.load(os.path.join(model_dir, model_name[0]))
        self.SNet = torch.load(os.path.join(model_dir, model_name[1]))

    def forward(self, x):
        r = self.DNet(x)
        d = x-r
        s = self.SNet(r, d)
        out = s+d
        return out, s, d

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

class SNet_jfver1(nn.Module):
    def __init__(self, kernel_size=3, image_channels=1):
        super(SNet_jfver1, self).__init__()
        layers = []
        self.convx = conv1_layers()
        self.convg = conv1_layers()
        self.conv2 = conv2_layers(2)

    def forward(self, x, g):
        x = self.convx(x)
        g = self.convg(g)

        out = self.conv2(torch.cat((x,g),1))
        return out

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
        self.conv1 = conv2_layers(2)
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

class SNet_texture_ver1(nn.Module):
    def __init__(self, kernel_size=3, image_channels=1):
        super(SNet_texture_ver1, self).__init__()
        layers = []
        self.conv1 = conv1_layers()
        self.tpn = TPN()
        self.conv2 = conv2_layers(2)

    def forward(self, x, g):
        g_ = self.conv1(g)
        x_ = self.tpn(x)

        out = self.conv2(torch.cat((x_,g_),1))
        return out

class SNet_texture_ver2(nn.Module):
    def __init__(self, kernel_size=3, image_channels=1):
        super(SNet_texture_ver2, self).__init__()
        layers = []
        self.filter1 = dynamic_filter(image_channels=image_channels)
        self.filter2 = dynamic_filter(image_channels=image_channels)
        self.filter3 = dynamic_filter(image_channels=image_channels)
        self.filter4 = dynamic_filter(image_channels=image_channels)

        self.filter_g = dynamic_filter(image_channels=image_channels)

        self.conv1 = conv1_layers()
        self.tpn = TPN()
        self.conv2 = conv2_layers(17)

    def forward(self, x, g):
        N, C, h, w = x.size()

        size1 = x
        size2 = F.interpolate(x, size=(h // 2, w // 2))
        size4 = F.interpolate(x, size=(h // 4, w // 4))
        size8 = F.interpolate(x, size=(h // 8, w // 8))

        f1 = self.filter1(size1)
        f2 = self.filter2(size2)
        f4 = self.filter3(size4)
        f8 = self.filter4(size8)

        g_ = self.filter_g(g)

        patches = F.pad(size1, (4,4,4,4), "constant", 0)
        patches = patches.unfold(2,9,1).unfold(3,9,1)
        patches = patches.permute(0,1,4,5,2,3).squeeze(1).reshape(-1,9*9,x.size(2),x.size(3))
        size1 = torch.sum(patches*f1, 1, keepdim=True)

        patches = F.pad(size2, (4,4,4,4), "constant", 0)
        patches = patches.unfold(2,9,1).unfold(3,9,1)
        patches = patches.permute(0,1,4,5,2,3).squeeze(1).reshape(-1,9*9,x.size(2),x.size(3))
        size2 = torch.sum(patches*f2, 1, keepdim=True)

        patches = F.pad(size4, (4,4,4,4), "constant", 0)
        patches = patches.unfold(2,9,1).unfold(3,9,1)
        patches = patches.permute(0,1,4,5,2,3).squeeze(1).reshape(-1,9*9,x.size(2),x.size(3))
        size4 = torch.sum(patches*f3, 1, keepdim=True)

        patches = F.pad(size8, (4,4,4,4), "constant", 0)
        patches = patches.unfold(2,9,1).unfold(3,9,1)
        patches = patches.permute(0,1,4,5,2,3).squeeze(1).reshape(-1,9*9,x.size(2),x.size(3))
        size8 = torch.sum(patches*f4, 1, keepdim=True)

        patches = F.pad(g, (4,4,4,4), "constant", 0)
        patches = patches.unfold(2,9,1).unfold(3,9,1)
        patches = patches.permute(0,1,4,5,2,3).squeeze(1).reshape(-1,9*9,x.size(2),x.size(3))
        g = torch.sum(patches*g_, 1, keepdim=True)

        size2 = F.interpolate(size2, size=(h, w))
        size4 = F.interpolate(size4, size=(h, w))
        size8 = F.interpolate(size8, size=(h, w))

        concat = torch.cat((size1, size2, size4, size8, g), 1)

        out = self.conv2(concat)
        return out
