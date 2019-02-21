import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
import cv2

class content_loss(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(content_loss, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = torch.cat((x, x, x), 1)
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3

class gan_loss(nn.Module):
    def __init__(self, image_channels=1, w=40):
        super(gan_loss, self).__init__()
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True))
        layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(128, eps=0.0001, momentum = 0.95))
        layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(128, eps=0.0001, momentum = 0.95))
        layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(128, eps=0.0001, momentum = 0.95))
        layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(256, eps=0.0001, momentum = 0.95))
        layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(256, eps=0.0001, momentum = 0.95))
        layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(512, eps=0.0001, momentum = 0.95))
        layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(512, eps=0.0001, momentum = 0.95))
        layers.append(nn.LeakyReLU(inplace=True))

        self.gan = nn.Sequential(*layers)

        self.fc1 = nn.Linear(in_features=512*3*3, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1)

        self._initialize_weights()

    def forward(self, x):
        x = self.gan(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        out = self.fc2(x)
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
            elif isinstance(m, nn.Linear):
                init.xavier_uniform(m.weight)
                print('init weight')
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
