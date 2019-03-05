from train import pretrain_SNet

if __name__ == '__main__':
    pretrain_SNet(lr=1e-3,device="cuda:0", model_name=['DNet(sigma=25)_2.pth'], save_name='SNet(25_denoised)_2.pth', guidance='denoised')
