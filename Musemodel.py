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
            nn.ConvTranspose2d(100, 1024,kernel_size=(3,1), stride = (1,1), padding = (0,0), bias=False),
            nn.ReLU(inplace = True),
            # state: (1000,3,1)
            nn.ConvTranspose2d(1024 , 512,kernel_size = 4, stride = 4, padding = 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            # state: (500,12,4)
            nn.ConvTranspose2d(512, 256,kernel_size = 4, stride = 4, padding = 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            # state: (200,48,16)
            nn.ConvTranspose2d(256, 100,kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace = True),
            # state: (100,96,32)
            nn.ConvTranspose2d( 100 , 2, kernel_size=3, stride=1, padding=1, bias=False),
            # state: (2,96,32)
            nn.Tanh()
        )

    def forward(self, x):
        tmp = self.model(x)
        output = tmp/2.0+0.5
        return output

class Generator_bar(nn.Module):
    def __init__(self, songsize = length_Section * length_tab):
        super(Generator_bar, self).__init__()

        self.model = nn.Sequential(
            # state : (2,24,32)
            nn.ConvTranspose2d(2, 100,kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.ReLU(inplace = True),
            # state: (100,48,64)
            nn.ConvTranspose2d(100 , 64,kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            # state: (64,96,128)
            nn.ConvTranspose2d(64, 16,kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),
            # state: (16,96,128)
            nn.ConvTranspose2d( 16 , 2, kernel_size=3, stride=1, padding=1, bias=False),
            # state: (2,384,128)
            nn.Tanh()
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
    z = torch.randn(100, 100, 1, 1)
    z = Variable(z).to(device)
    fake_song_tmp = G_tmp_model(z)
    print(fake_song_tmp.shape)
    p1 = fake_song_tmp[:,:,0:24,:]
    print("p1 :",p1.shape)
    p2 = fake_song_tmp[:,:,24:48,:]
    print("p2 :",p2.shape)
    p3 = fake_song_tmp[:,:,48:72,:]
    p4 = fake_song_tmp[:,:,72:96,:]
    fake_song_p1 = G_bar_model(p1)
    print("fake_song_p1 :",fake_song_p1.shape)
    fake_song_p2 = G_bar_model(p2)
    fake_song_p3 = G_bar_model(p3)
    fake_song_p4 = G_bar_model(p4)
    fake_song = torch.cat((fake_song_p1,fake_song_p2,fake_song_p3,fake_song_p4),2)
    print("fake_song : ",fake_song.shape)
    outputs_dis = D_model(fake_song)
    print(outputs_dis.shape)

def print_model():
    G_tmp_model = Generator_tmp()
    G_bar_model = Generator_bar()
    D_model = Discriminator()
    print(D_model)

if __name__ == '__main__':
    print_model()
