from train import *

if __name__ == '__main__':
    train(batch_size=128, n_epoch=100, sigma=25, lr=1e-4, device="cuda:0", model_name=['DNet(sigma=25)_1.pth', 4])
