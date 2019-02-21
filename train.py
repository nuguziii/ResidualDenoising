from model import DNet, JointNet, Model
from loss_model import content_loss, gan_loss
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

class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

def bce_loss(input, target):
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

class discriminator_loss(_Loss):  # PyTorch 0.4.1
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(discriminator_loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        N = target.size()
        true_labels = Variable(torch.ones(N)).type(input.type())
        real_image_loss = bce_loss(target, true_labels)
        fake_image_loss = bce_loss(input, 1 - true_labels)
        loss = real_image_loss + fake_image_loss
        return loss

def generator_loss(input):
    N = input.size()
    true_labels = Variable(torch.ones(N)).type(input.type())
    loss = bce_loss(input, true_labels)
    return loss

class perceptual_loss(_Loss):
    """
    Definition: perceptual_loss = vgg loss + gan loss + mse loss
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum', device="cuda:0"):
        super(perceptual_loss, self).__init__(size_average, reduce, reduction)
        self.d = gan_loss()
        self.vgg = content_loss()
        if torch.cuda.is_available():
            self.d = self.d.to(device)
            self.vgg = self.vgg.to(device)

    def forward(self, input, target):
        optimizer = optim.Adam(self.d.parameters(), lr=1e-3)
        criterion = discriminator_loss()
        d_loss = criterion(self.d(input),self.d(target))
        optimizer.step()

        mse_loss = torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum')
        gan_loss = generator_loss(self.d(input))
        vgg_loss = torch.nn.functional.mse_loss(self.vgg(input)[3], self.vgg(target)[3], size_average=None, reduce=None, reduction='sum')
        return mse_loss+1e-3*gan_loss+2e-6*vgg_loss, d_loss

def train(batch_size=128, n_epoch=100, sigma=25, lr=1e-3, device="cuda:0", data_dir='./data/Train400', model_dir='models', model_name=None, save_name=None, discription=None):
    device = torch.device(device)

    print('--\t', discription)
    print('--\t epoch %4d batch_size %4d sigma %4d' % (n_epoch, batch_size, sigma))

    model = Model(model_dir=model_dir,model_name=model_name, guidance='denoised', kernel_size=kernel_size) #guidance='noisy, denoised'

    model.train()

    criterion = perceptual_loss(device)

    if torch.cuda.is_available():
        model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    for epoch in range(n_epoch):
        x = dg.datagenerator(data_dir=data_dir).astype('float32')/255.0
        x = torch.from_numpy(x.transpose((0, 3, 1, 2)))
        dataset = DenoisingDataset(x, sigma)
        loader = DataLoader(dataset=dataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()
        n_count=0
        for cnt, batch_yx in enumerate(loader):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                batch_original, batch_noise = batch_yx[1].to(device), batch_yx[0].to(device)
            loss, d_loss = criterion(model(batch_noise), batch_original)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if cnt%100 == 0:
                print('%4d %4d / %4d g_loss = %2.4f\t(d_loss = %2.4f)' % (epoch+1, cnt, x.size(0)//batch_size, loss.item()/batch_size, d_loss.item()/batch_size))
            n_count +=1

        elapsed_time = time.time() - start_time
        print('epoch = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
        torch.save(model, os.path.join(model_dir, save_name))

    torch.save(model, os.path.join(model_dir, save_name))

def pretrain_SNet(batch_size=128, n_epoch=100, sigma=25, lr=1e-3, device="cuda:0", data_dir='./data/Train400', model_dir='models', model_name=None, save_name=None, guidance='denoised'):
    device = torch.device(device)

    print('\n\n')
    print('--\t This model is pre-trained SNet saved as ',save_name )
    print('--\t epoch %4d batch_size %4d sigma %4d' % (n_epoch, batch_size, sigma))

    DNet = torch.load(os.path.join(model_dir, model_name[0]))
    model = JointNet(kernel_size=3) #guidance='noisy, denoised'

    DNet.eval()
    model.train()

    criterion = sum_squared_error()

    if torch.cuda.is_available():
        DNet.to(device)
        model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    for epoch in range(n_epoch):
        x = dg.datagenerator(data_dir=data_dir).astype('float32')/255.0
        x = torch.from_numpy(x.transpose((0, 3, 1, 2)))
        dataset = DenoisingDataset(x, sigma)
        loader = DataLoader(dataset=dataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()
        n_count=0
        for cnt, batch_yx in enumerate(loader):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                batch_original, batch_noise = batch_yx[1].to(device), batch_yx[0].to(device)

            r = DNet(batch_noise)
            d = batch_noise-r
            r=1.55*(r+0.5)-0.8
            if guidance is 'noisy':
                s = model(r, batch_noise)
            elif guidance is 'denoised':
                s = model(r, d)
            target = 1.8*(batch_original-d+0.5)-0.8

            loss = criterion(s, target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if cnt%100 == 0:
                print('%4d %4d / %4d loss = %2.4f' % (epoch+1, cnt, x.size(0)//batch_size, loss.item()/batch_size))
            n_count +=1

        elapsed_time = time.time() - start_time
        print('epoch = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
        torch.save(model, os.path.join(model_dir, save_name))

    torch.save(model, os.path.join(model_dir, save_name))

if __name__ == '__main__':
    pretrain_SNet(device="cuda:0", model_name=['DNet_sigma=25_1.pth'], save_name='SNet_25_denoised.pth', guidance='denoised')
