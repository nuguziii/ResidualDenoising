import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
import cv2

class content_loss(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(content_loss, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()

        for x in range(8):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(8, 35):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = torch.cat((x, x, x), 1)
        h = self.slice1(x)
        vgg22 = h
        h = self.slice2(h)
        vgg54 = h
        return vgg22, vgg54

class discriminator(nn.Module):
    def __init__(self, image_channels=1, w=40):
        super(discriminator, self).__init__()
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

        self.fc1 = nn.Linear(in_features=512*5*5, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.gan(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.sig(x)
        return out
