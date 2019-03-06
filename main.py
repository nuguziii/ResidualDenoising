from train import pretrain_SNet, pretrain_DNet

if __name__ == '__main__':
    pretrain_DNet(device="cuda:0", save_name='DNet(sigma=25)_1.pth')
    pretrain_DNet(depth=7, device="cuda:0", save_name='DNet(sigma=25)_2.pth')
