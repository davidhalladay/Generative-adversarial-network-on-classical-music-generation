import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from pypianoroll import Multitrack, Track
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import skimage.io
import skimage
import os
import time
import pandas as pd
import random
import pickle
import model
from Dataset import Dataset
from data_preprocess import make_mid

random.seed(312)
torch.manual_seed(312)

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def main():
    # parameters
    file_root = "./raw_classical_music_data"
    learning_rate = 0.008
    num_epochs = 401
    batch_size = 100
    input_dim = 100

    # fixed input for model eval
    # rand_inputs = (5,1000,1,1) , pair_inputs = (5,1000,1,1)
    with torch.no_grad():
        rand_inputs = Variable(torch.randn(1,input_dim, 1, 1))

    # create the save log file
    print("Create the directory")
    if not os.path.exists("./save"):
        os.makedirs("./save")
    if not os.path.exists("./logfile"):
        os.makedirs("./logfile")
    if not os.path.exists("./logfile/GAN"):
        os.makedirs("./logfile/GAN")
    if not os.path.exists("./save_songs"):
        os.makedirs("./save_songs")
    if not os.path.exists("./save_songs/GAN"):
        os.makedirs("./save_songs/GAN")

    # load my Dataset
    train_dataset = Dataset(filepath = file_root,num = 1000)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=1)

    print('the dataset has %d size.' % (len(train_dataset)))
    print('the batch_size is %d' % (batch_size))

    # models setting
    G_tmp_model = model.Generator_tmp()
    G_bar_model = model.Generator_bar()
    D_model = model.Discriminator()

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        rand_inputs = rand_inputs.to(device)
        G_tmp_model = G_tmp_model.to(device)
        G_bar_model = G_bar_model.to(device)
        D_model = D_model.to(device)

    # setup optimizer
    G_tmp_optimizer = optim.Adam(G_tmp_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    G_bar_optimizer = optim.Adam(G_bar_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    criterion_dis = nn.BCELoss()
    criterion_att = nn.BCELoss()

    D_loss_list = []
    G_loss_list = []
    D_real_acc_list = []
    D_fake_acc_list = []
    D_Real_att_loss_list = []
    D_Fake_att_loss_list = []

    print("Starting training...")

    for epoch in range(num_epochs):
        G_tmp_model.train()
        G_bar_model.train()
        D_model.train()

        print("Epoch:", epoch+1)
        epoch_D_loss = 0.0
        epoch_G_loss = 0.0
        D_real_total_acc = 0.0
        D_fake_total_acc = 0.0
        D_Real_att_total_loss = 0.0
        D_Fake_att_total_loss = 0.0

        if (epoch+1) == 11:
            G_tmp_optimizer.param_groups[0]['lr'] /= 2
            G_bar_optimizer.param_groups[0]['lr'] /= 2
            D_optimizer.param_groups[0]['lr'] /= 2

        if (epoch+1) == 20:
            G_tmp_optimizer.param_groups[0]['lr'] /= 2
            G_bar_optimizer.param_groups[0]['lr'] /= 2
            D_optimizer.param_groups[0]['lr'] /= 2

        if (epoch+1) == 60:
            G_tmp_optimizer.param_groups[0]['lr'] /= 2
            G_bar_optimizer.param_groups[0]['lr'] /= 2
            D_optimizer.param_groups[0]['lr'] /= 2

        if (epoch+1) == 100:
            G_tmp_optimizer.param_groups[0]['lr'] /= 2
            G_bar_optimizer.param_groups[0]['lr'] /= 2
            D_optimizer.param_groups[0]['lr'] /= 2

        if (epoch+1) == 150:
            G_tmp_optimizer.param_groups[0]['lr'] /= 2
            G_bar_optimizer.param_groups[0]['lr'] /= 2
            D_optimizer.param_groups[0]['lr'] /= 2

        for i, data in enumerate(train_loader):
            batch_size = len(data)
            real_labels = torch.ones(batch_size)
            fake_labels = torch.zeros(batch_size)
            data = Variable(data).to(device)
            real_labels = Variable(real_labels).to(device)
            fake_labels = Variable(fake_labels).to(device)

            # train the Generator tmp
            G_tmp_model.zero_grad()
            G_bar_model.zero_grad()
            z = torch.randn(batch_size, input_dim, 1, 1)
            z = Variable(z).to(device)
            fake_song_tmp = G_tmp_model(z)
            fake_song = G_bar_model(fake_song_tmp)
            outputs_dis = D_model(fake_song)
            G_loss = criterion_dis(outputs_dis,real_labels)
            epoch_G_loss += G_loss.item()
            G_loss.backward()
            G_tmp_optimizer.step()
            G_bar_optimizer.step()

            # train the Discriminator
            # BCE_Loss(x, y) = - y * log(D(x)) - (1-y) * log(1 - D(x))
            # real images , real_labels == 1

            D_model.zero_grad()
            ####################################################
            # self test
            print("num of non-zero :",np.sum(data[0].cpu().data.numpy() != 0.))
            #print(data[0].cpu().data.numpy())
            #data = data[0].cpu().data.numpy()
            ## torch.Size([1, 2, 384, 128])
            #data = data.transpose(1, 2, 0)
            #tarck1 = data[:,:,0]
            #tarck2 = data[:,:,1]
            #tarck1 = Track(tarck1, program=0, is_drum=False, name='unknown')
            #tarck2 = Track(tarck2, program=0, is_drum=False, name='unknown')
            #cc = Multitrack(tracks=[tarck1,tarck2], tempo=120.0,beat_resolution=24)
            #cc.write('./my_test')
            ####################################################
            outputs_dis = D_model(data)
            D_real_loss = criterion_dis(outputs_dis, real_labels)
            D_real_acc = np.mean((outputs_dis > 0.5).cpu().data.numpy())

            # fake images
            # First term of the loss is always zero since fake_labels == 0
            # we don't want to colculate the G gradient
            outputs_dis = D_model(fake_song.detach())
            D_fake_loss = criterion_dis(outputs_dis, fake_labels)
            D_fake_acc = np.mean((outputs_dis < 0.5).cpu().data.numpy())
            D_loss = (D_real_loss + D_fake_loss) / 2.

            D_loss.backward()
            D_optimizer.step()

            D_real_total_acc += D_real_acc
            D_fake_total_acc += D_fake_acc

            epoch_D_loss += D_loss.item()
            print('Epoch [%d/%d], Iter [%d/%d] G loss %.4f, D loss %.4f , LR = %.6f'
            %(epoch+1, num_epochs, i+1, len(train_loader), G_loss.item(), D_loss.item(), learning_rate))

        if (epoch) % 50 == 0:
            save_checkpoint('./save/GAN-G-tmp-%03i.pth' % (epoch) , G_tmp_model, G_tmp_optimizer)
            save_checkpoint('./save/GAN-G-bar-%03i.pth' % (epoch) , G_bar_model, G_bar_optimizer)
            save_checkpoint('./save/GAN-D-%03i.pth' % (epoch) , D_model, D_optimizer)

        # save loss data
        print("training D Loss:", epoch_D_loss/len(train_loader.dataset))
        print("training G Loss:", epoch_G_loss/len(train_loader.dataset))
        D_loss_list.append(epoch_D_loss/len(train_loader.dataset))
        G_loss_list.append(epoch_G_loss/len(train_loader.dataset))
        D_real_acc_list.append(D_real_total_acc/len(train_loader))
        D_fake_acc_list.append(D_fake_total_acc/len(train_loader))

        # testing
        if (epoch) % 20 == 0:
            G_tmp_model.eval()
            G_bar_model.eval()
            tmp_output = G_tmp_model(rand_inputs)
            test_output = G_bar_model(tmp_output)
            test_song = test_output.cpu().data.numpy()
            print(test_song)
            print("test_song num of non-zero :",np.sum(test_song >= 0.2))
            # torch.Size([1, 2, 384, 128])
            test_song = test_song.transpose(0, 2, 3, 1)
            tarck1,tarck2 = make_mid(test_song, 0.2)
            print(tarck1)
            cc = Multitrack(tracks=[tarck1,tarck2], tempo=120.0,beat_resolution=24)
            cc.write('./save_songs/GAN/%03d' %(epoch+1))

        # epoch done
        print('-'*88)

    # shuffle
if __name__ == '__main__':
    main()
