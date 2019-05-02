import data_generator as dg
from data_generator import DenoisingDataset

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time, os
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import random
import torch.nn.init as init
import torch.nn.functional as F

class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum')

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
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

    def forward(self, x):
        r = self.dncnn(x)
        return r

def train(batch_size=128, n_epoch=150, sigma=25, lr=1e-3, lr2=1e-5, depth=17, device="cuda:0", data_dir='./data/Train400', model_dir='models'):
    device = torch.device(device)

    from datetime import date
    model_dir = os.path.join(model_dir,"result_"+"".join(str(date.today()).split('-')[1:]))
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    save_name = "net.pth"
    save_name2 = "net2.pth"

    print('\n')
    print('--\t This model is pre-trained DNet saved as ',save_name )
    print('--\t epoch %4d batch_size %4d sigma %4d depth %4d' % (n_epoch, batch_size, sigma, depth))
    print('\n')

    model = DnCNN(depth=depth)
    model2 = DnCNN(depth=depth)

    model.train()
    model2.train()

    criterion = sum_squared_error()
    criterion_l1 = nn.L1Loss(size_average=None, reduce=None, reduction='sum')

    if torch.cuda.is_available():
        model.to(device)
        model2.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    optimizer2 = optim.Adam(model2.parameters(), lr=lr2, weight_decay=1e-5)
    scheduler2 = MultiStepLR(optimizer2, milestones=[30, 60, 90], gamma=0.2)  # learning rates

    for epoch in range(n_epoch):
        x = dg.datagenerator(data_dir=data_dir).astype('float32')/255.0
        x = torch.from_numpy(x.transpose((0, 3, 1, 2)))

        dataset=None
        dataset = DenoisingDataset(x, sigma)

        loader = DataLoader(dataset=dataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        epoch_loss_first = 0
        start_time = time.time()
        n_count=0
        for cnt, batch_yx in enumerate(loader):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                batch_original, batch_noise= batch_yx[1].to(device), batch_yx[0].to(device)

            residual = model(batch_noise)
            loss_first = criterion(batch_noise-residual, batch_original)
            loss_first.backward(retain_graph=True)
            optimizer.step()

            structure_residual = model2(residual)
            target = batch_original - (batch_noise - residual)
            structure = residual - structure_residual
            loss = criterion_l1(structure, target)
            loss.backward()
            optimizer2.step()

            epoch_loss_first += loss_first.item()
            epoch_loss += loss.item()

            if cnt%100 == 0:
                print('%4d %4d / %4d 1_loss = %2.4f loss = %2.4f' % (epoch+1, cnt, x.size(0)//batch_size, loss_first.item()/batch_size, loss.item()/batch_size))
            n_count +=1

        elapsed_time = time.time() - start_time
        print('epoch = %4d , sigma = %4d, 1_loss = %4.4f, loss = %4.4f , time = %4.2f s' % (epoch+1, sigma, epoch_loss_first/n_count, epoch_loss/n_count, elapsed_time))
        if (epoch+1)%25 == 0:
            torch.save(model, os.path.join(model_dir, save_name.replace('.pth', '_epoch%03d.pth') % (epoch+1)))
            torch.save(model2, os.path.join(model_dir, save_name2.replace('.pth', '_epoch%03d.pth') % (epoch+1)))

    torch.save(model, os.path.join(model_dir, save_name))
    torch.save(model2, os.path.join(model_dir, save_name2))

if __name__ == '__main__':
    train(batch_size=128, n_epoch=150, sigma=50, lr=1e-3, lr2=1e-5, depth=17, device="cuda:0", data_dir='../data/Train400', model_dir='models')
