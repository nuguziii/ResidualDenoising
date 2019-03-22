from model import *
from loss_model import *
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
        self.vgg = content_loss()
        if torch.cuda.is_available():
            self.vgg = self.vgg.to(device)

    def forward(self, input, target, errG):
        mse_loss = torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)
        gan_loss = errG
        vgg_loss = torch.nn.functional.mse_loss(self.vgg(input)[1], self.vgg(target)[1], size_average=None, reduce=None, reduction='sum').div_(2)
        #return mse_loss+1e-3*gan_loss+2e-6*vgg_loss
        return mse_loss, gan_loss, vgg_loss

def initialize_weights(self):
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


def train(batch_size=128, n_epoch=100, sigma=25, lr=1e-4, device="cuda:0", data_dir='./data/Train400', model_dir='models', model_name=None):
    device = torch.device(device)

    if not os.path.exists(os.path.join(model_dir,"model"+str(sigma)+"m"+str(model_name[1]))):
        os.mkdir(os.path.join(model_dir,"model"+str(sigma)+"m"+str(model_name[1])))

    from datetime import date
    save_name = "model_mode" + str(model_name[1])+"_"+ "".join(str(date.today()).split('-')[1:]) + ".pth"

    f = open(os.path.join(model_dir,save_name.replace(".pth",".txt")),'w')

    f.write(('--\t This is end to end model saved as '+ save_name+'\n'))
    f.write(('--\t epoch %4d batch_size %4d sigma %4d\n' % (n_epoch, batch_size, sigma)))
    f.write(model_name[0])

    modelG = Model(model_dir=model_dir,model_name=model_name) #guidance='noisy, denoised'

    print(modelG)
    f.write(str(modelG))
    f.write('\n\n')

    ngpu = 2
    if (device.type == 'cuda') and (ngpu > 1):
        modelG = nn.DataParallel(modelG, list(range(ngpu)))

    modelG.apply(initialize_weights)

    '''
    modelD = discriminator()
    if (device.type == 'cuda') and (ngpu > 1):
        modelD = nn.DataParallel(modelD, list(range(ngpu)))
    modelD.apply(initialize_weights)
    '''

    criterionG = perceptual_loss(device)
    criterion = nn.BCELoss()
    criterionS = sum_squared_error()

    if torch.cuda.is_available():
        modelG.to(device)
        #modelD.to(device)

    optimizerG = optim.Adam(modelG.parameters(), lr=lr, betas=(0.5, 0.999),  weight_decay=1e-5)
    #optimizerD = optim.Adam(modelD.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
    scheduler = MultiStepLR(optimizerG, milestones=[30, 60, 90], gamma=0.2)  # learning rates

    for epoch in range(n_epoch):
        x = dg.datagenerator(data_dir=data_dir).astype('float32')/255.0
        x = torch.from_numpy(x.transpose((0, 3, 1, 2)))
        dataset = DenoisingDataset(x, sigma)
        loader = DataLoader(dataset=dataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss_g = 0
        start_time = time.time()
        n_count=0
        for cnt, batch_yx in enumerate(loader):
            if torch.cuda.is_available():
                batch_original, batch_noise = batch_yx[1].to(device), batch_yx[0].to(device)
            '''
            modelD.zero_grad()
            b_size = batch_original.size(0)
            label = torch.full((b_size,), 1, device=device)
            output = modelD(batch_original).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward(retain_graph=True)

            fake, structure, denoised = modelG(batch_noise)
            label.fill_(0)
            output = modelD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward(retain_graph=True)

            d_loss = errD_real + errD_fake
            optimizerD.step()
            '''
            modelG.zero_grad()
            fake, structure, denoised = modelG(batch_noise)
            label = torch.full((batch_original.size(0),), 1, device=device)
            #output = modelD(fake).view(-1)
            #errG = criterion(output, label)

            s_loss = criterionS(structure, batch_original-denoised)
            s_loss.backward(retain_graph=True)

            mse_loss, gan_loss, vgg_loss= criterionG(fake, batch_original, 0)
            g_loss = mse_loss+2e-4*vgg_loss
            g_loss.backward(retain_graph=True)
            epoch_loss_g += g_loss.item()
            optimizerG.step()

            if cnt%100 == 0:
                line = '%4d %4d / %4d g_loss = %2.4f\t(s_loss = %2.4f)' % (epoch+1, cnt, x.size(0)//batch_size, g_loss.item()/batch_size, s_loss.item()/batch_size)
                print(line)
                print("mse_loss=%4d / vgg_loss=%4d" % (mse_loss.item()/batch_size, vgg_loss.item()/batch_size))
                f.write(line)
                f.write('\n')
            n_count +=1

        elapsed_time = time.time() - start_time
        line = 'epoch = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss_g/(n_count*batch_size), elapsed_time)
        print(line)
        f.write(line)
        f.write('\n')
        if (epoch+1)%20 == 0:
            torch.save(modelG, os.path.join(model_dir, save_name.replace('.pth', '_epoch%03d.pth') % (epoch+1)))

    torch.save(modelG, os.path.join(model_dir, save_name))
    f.close()

def pretrain_SNet(batch_size=128, n_epoch=100, sigma=25, lr=1e-4, device="cuda:0", data_dir='./data/Train400', model_dir='models/SNet', model_name=None, model=0):
    device = torch.device(device)
    if not os.path.exists(model_dir):
        os.mkdir(os.path.join(model_dir))

    DNet = torch.load(os.path.join(model_dir, model_name[0]))
    if model==0:
        model = SNet_jfver1()
        save_name = 'SNet_jfver1'
    elif model==1:
        model = SNet_dfver1()
        save_name = 'SNet_dfver1'
    elif model==2:
        model = SNet_dfver2()
        save_name = 'SNet_dfver2'
    elif model==3:
        model = SNet_texture_ver1()
        save_name = 'SNet_texture_ver1'

    from datetime import date
    save_name = save_name + "_"+ "".join(str(date.today()).split('-')[1:]) + ".pth"

    f = open(os.path.join(model_dir,save_name.replace(".pth",".txt")),'w')

    print('\n')
    print('--\t This model is pre-trained SNet saved as ',save_name )
    print('--\t epoch %4d batch_size %4d sigma %4d' % (n_epoch, batch_size, sigma))
    print('\n')

    DNet.eval()
    model.train()

    print(model)
    print("\n")

    criterion = sum_squared_error()

    if torch.cuda.is_available():
        DNet.to(device)
        model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
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
            #r=1.55*(r+0.5)-0.8
            s = model(r, d)
            #target = 1.8*(batch_original-d+0.5)-0.8
            loss = criterion(s, batch_original-d)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if cnt%100 == 0:
                line = '%4d %4d / %4d loss = %2.4f' % (epoch+1, cnt, x.size(0)//batch_size, loss.item()/batch_size)
                print(line)
                f.write(line)
            n_count +=1

        elapsed_time = time.time() - start_time
        line = 'epoch = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time)
        print(line)
        f.write(line)
        if (epoch+1)%1 == 0:
            torch.save(model, os.path.join(model_dir, save_name.replace('.pth', '_epoch%03d.pth') % (epoch+1)))

    torch.save(model, os.path.join(model_dir, save_name))
    f.close()

def pretrain_DNet(batch_size=128, n_epoch=150, sigma=25, lr=1e-3, depth=17, device="cuda:0", data_dir='./data/Train400', model_dir='models'):
    device = torch.device(device)

    model_dir = os.path.join(model_dir,"DNet_s"+str(sigma)+"d"+str(depth))
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    from datetime import date
    save_name = "DNet_s"+str(sigma)+"d"+str(depth)+"_"+ "".join(str(date.today()).split('-')[1:]) + ".pth"

    print('\n')
    print('--\t This model is pre-trained DNet saved as ',save_name )
    print('--\t epoch %4d batch_size %4d sigma %4d depth %4d' % (n_epoch, batch_size, sigma, depth))
    print('\n')

    model = DNet(depth=depth)

    model.train()

    print(model)
    print("\n")

    criterion = sum_squared_error()

    if torch.cuda.is_available():
        model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    for epoch in range(n_epoch):
        x = dg.datagenerator(data_dir=data_dir).astype('float32')/255.0
        x = torch.from_numpy(x.transpose((0, 3, 1, 2)))

        dataset=None
        dataset = DenoisingDataset(x, sigma)

        loader = DataLoader(dataset=dataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()
        n_count=0
        for cnt, batch_yx in enumerate(loader):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                batch_original, batch_noise= batch_yx[1].to(device), batch_yx[0].to(device)

            r = model(batch_noise)
            loss = criterion(batch_noise-r, batch_original)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if cnt%100 == 0:
                print('%4d %4d / %4d loss = %2.4f' % (epoch+1, cnt, x.size(0)//batch_size, loss.item()/batch_size))
            n_count +=1

        elapsed_time = time.time() - start_time
        print('epoch = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
        if (epoch+1)%10 == 0:
            torch.save(model, os.path.join(model_dir, save_name.replace('.pth', '_epoch%03d.pth') % (epoch+1)))

    torch.save(model, os.path.join(model_dir, save_name))
if __name__ == '__main__':
    pretrain_SNet(device="cuda:0", model_name=['DNet_sigma=25_1.pth'], save_name='SNet_25_denoised.pth', guidance='denoised')
