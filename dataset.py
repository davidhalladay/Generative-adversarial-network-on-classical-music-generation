import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import os.path
import sys
import string
import pandas as pd
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from data_preprocess import *

class Dataset(Dataset):
    def __init__(self, filepath,num):
        self.length_Section = 96
        self.length_tab = 4
        self.num = num
        print("Load file from :" ,filepath)
        self.song = load_file(filepath)
        self.song , self.label = make_data(self.song)
        rand_ch = np.random.choice(self.song.shape[0],num,replace=False)
        self.song = self.song[rand_ch]
        self.label = torch.FloatTensor(self.label).view(-1, 1, 1, 1)
        # song = (7803, 384, 128, 2)
        self.song = self.song.transpose(0, 3, 1, 2)
        # song = (7803, 2, 384, 128)
        self.song = torch.FloatTensor(self.song)
        print("")
        print("Loading file completed.")

        self.num_samples = len(self.song)

    def __getitem__(self, index):
        data = self.song[index]
        # label = self.label[index]
        return data #, label

    def __len__(self):
        return self.num_samples

def main():
    file_root = "raw_classical_music_data"
    train_dataset = Dataset(filepath = file_root,num = 1000)
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=0)
    train_iter = iter(train_loader)
    print(len(train_loader.dataset))
    print(len(train_loader))
    for epoch in range(1):
        song = next(train_iter)
        print(song.shape)
        print(song)
        print("num of 1:",np.sum(np.array(song)!=0.))
        # [1, 2, 128, 960]
        song = np.array(song)
        song = song.transpose(0, 3, 2, 1)
        track1 = Track(song[0,:,:,0], program=0, is_drum=False, name='unknown')
        track2 = Track(song[0,:,:,1], program=0, is_drum=False, name='unknown')
        cc = Multitrack(tracks=[track1,track2], tempo=120.0, beat_resolution=24, name='unknown')
        cc.write('test_1.mid')
# main()

if __name__ == '__main__':
    main()
