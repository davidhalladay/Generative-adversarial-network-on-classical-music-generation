########################################################
## Author : Wan-Cyuan Fan
## Reference : https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
########################################################

import torch
import torch.nn as nn
from torch.autograd import Variable

length_Section = 96
length_tab = 4

class Generator_tmp(nn.Module):
    def __init__(self, songsize = length_Section * length_tab):
        super(Generator_tmp, self).__init__()

        self.model = nn.Sequential(
            # state : (100,1,1)
            nn.ConvTranspose2d(100, 200 ,kernel_size=(4,2), stride = (1,1), padding = (0,0), bias=False),
            nn.ReLU(inplace = True),
            # state: (200,4,2)
            nn.ConvTranspose2d(200 , 100,kernel_size=(3,4), stride = (3,2), padding = (0,1), bias=False),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace = True),
            # state: (100,12,4)
            nn.ConvTranspose2d(100, 50,kernel_size=(4,4), stride = (2,2), padding = (1,1), bias=False),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace = True),
            # state: (50,24,8)
            nn.ConvTranspose2d(50 , 2, kernel_size=3, stride=1, padding=1, bias=False),
            # state: (2,24,8)
            nn.Tanh()
        )
        self.test = nn.Sequential(
            # state : (100,1,1)
            nn.ConvTranspose2d(100, 200 ,kernel_size=(4,2), stride = (1,1), padding = (0,0), bias=False),
            nn.ReLU(inplace = True),
            # state: (200,4,2)
            nn.ConvTranspose2d(200 , 100,kernel_size=(3,4), stride = (3,2), padding = (0,1), bias=False),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace = True),
            # state: (100,12,4)
            nn.ConvTranspose2d(100, 50,kernel_size=(4,4), stride = (2,2), padding = (1,1), bias=False),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace = True)
            # state: (50,24,8)
        )
    def forward(self, x):
        tmp = self.model(x)
        output = tmp/2.0+0.5
        return output

class Generator_bar(nn.Module):
    def __init__(self, songsize = length_Section * length_tab):
        super(Generator_bar, self).__init__()

        self.model = nn.Sequential(
            # state : (2,24,8)
            nn.ConvTranspose2d(2, 10,kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.ReLU(inplace = True),
            # state: (10,48,16)
            nn.ConvTranspose2d(10 , 20,kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace = True),
            # state: (40,96,32)
            nn.ConvTranspose2d(20, 30,kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(30),
            nn.ReLU(inplace = True),
            # state: (60,192,64)
            nn.ConvTranspose2d(30, 40,kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace = True),
            # state: (40,384,128)
            nn.ConvTranspose2d( 40 , 2, kernel_size=3, stride=1, padding=1, bias=False),
            # state: (2,384,128)
            nn.Tanh()
        )

        self.test = nn.Sequential(
            # state : (2,60,8)
            nn.ConvTranspose2d(2, songsize * 4,kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.ReLU(inplace = True),
            # state: (songsize * 4,120,16)

        )

    def forward(self, x):
        img = self.model(x)
        output = img/2.0+0.5
        return output


class Discriminator(nn.Module):
    def __init__(self, songsize = length_Section * length_tab):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # state: (2,384,128)
            nn.Conv2d(2, 20,kernel_size=(4,4), stride = (3,4), padding = (1,1), bias=False),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(0.2, inplace = True),
            # state: (20,128,32)
            nn.Conv2d(20, 40,kernel_size=(4,4), stride = (4,2), padding = (1,1), bias=False),
            nn.BatchNorm2d(40),
            nn.LeakyReLU(0.2, inplace = True),
            # state: (40,32,16)
            nn.Conv2d(40, 60,kernel_size=(4,4), stride = (4,2), padding = (1,1), bias=False),
            nn.BatchNorm2d(60),
            nn.LeakyReLU(0.2, inplace = True),
            # state: (60,8,8)
            nn.Conv2d(60, 80,kernel_size=(4,4), stride = (2,2), padding = (1,1), bias=False),
            nn.BatchNorm2d(80),
            nn.LeakyReLU(0.2, inplace = True)
            # state: (80,4,4)
        )
        self.dis = nn.Sequential(
            nn.Conv2d(80 , 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.att = nn.Sequential(
            nn.Conv2d(80 , 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        song = self.model(x)
        out_dis = self.dis(song)
        return out_dis

def test():
    G_tmp_model = Generator_tmp()
    G_bar_model = Generator_bar()
    D_model = Discriminator()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        G_tmp_model = G_tmp_model.to(device)
        G_bar_model = G_bar_model.to(device)
        D_model = D_model.to(device)

    # Fowarding a dummy tensor
    G_tmp_model.zero_grad()
    G_bar_model.zero_grad()
    x = torch.randn(100,100,1,1).to(device) # shape of N*C*H*W
    x = G_tmp_model(x)
    print(x.shape)
    x = G_bar_model(x)
    print(x.shape)
    x = D_model(x)
    print(x.shape)
    # torch.Size([100, 2, 24, 8])
    # torch.Size([100, 2, 384, 128])
    # torch.Size([100, 1, 1, 1])

if __name__ == '__main__':
    test()
