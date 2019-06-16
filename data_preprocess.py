import numpy as np
from pypianoroll import Multitrack, Track
from matplotlib import pyplot as plt
import os

# setting Global var.
length_Section = 96
length_tab = 4
num_of_song = [14, 3, 29, 7, 10, 9, 48, 9, 3, 16, 21, 16, 15, 21, 8, 29, 24, 12]

def load_file(root_path):
    # parser the data
    dir_list = os.listdir(root_path)
    dir_list.sort()
    file_name = []
    file_num = []
    for dir in dir_list:
        tmp_name = os.listdir(os.path.join("./raw_classical_music_data",dir))
        tmp_name.sort()
        file_name.append(tmp_name)
        file_num.append(len(tmp_name))
    print("Num of file in each dir : " ,file_num)

    # Parse a MIDI file to a `pypianoroll.Multitrack` instance
    data_list = []
    for i,dir in enumerate(dir_list):
        print("\rPhraser the data : %d/%d" %(i+1,len(dir_list)), end = "")
        for name in file_name[i]:
            file_path = os.path.join('./raw_classical_music_data',dir,name)
            tmp_pypianoroll = Multitrack(file_path)
            tmp_pypianoroll.remove_empty_tracks()
            tmp_pypianoroll.pad_to_multiple(length_Section*length_tab)
            tmp_np = tmp_pypianoroll.get_stacked_pianoroll()
            data_list.append(tmp_np)
    data_list = np.array(data_list)
    #np.save("all_data.npy",data_list)
    print("")
    print("finish praser the data...")
    return data_list

def count_tab(data_list):
    tab_size = []
    for i in range(data_list.shape[0]):
        tab_size.append(data_list[i].shape[0])
    return tab_size

def test():
    tmp_pypianoroll = Multitrack("./mz_545_1.mid")
    print(tmp_pypianoroll.tracks[0])
    tmp_pypianoroll.remove_empty_tracks()
    tmp_np = tmp_pypianoroll.get_stacked_pianoroll()

    tarck1 = tmp_np[:960,:,0]
    tarck2 = tmp_np[:960,:,1]
    tarck1 = tarck1 > 0.1
    tarck2 = tarck2 > 0.1

    tarck1 = Track(tarck1, program=0, is_drum=False, name='unknown')
    tarck2 = Track(tarck2, program=0, is_drum=False, name='unknown')


    cc = Multitrack(tracks=[tarck1,tarck2], tempo=120.0,beat_resolution=24)

    #print(cc)
    cc.write('test_2.mid')
    #cc = Multitrack(tmp_np)

def make_data(data):
    new_data = []
    # make label (musician)
    label = []
    for i,num in enumerate(num_of_song):
        for c in range(num):
            label.append(i)
    label = np.array(label)
    # make cutting data (cutting size = length_Section * length_tab)
    print("cutting data...")
    for idx in range(data.shape[0]):
        song = data[idx]  # song = [ num of m , 128 , 2 ]
        if song.shape[2] != 2:
            continue
        length_of_data = (length_Section*length_tab)
        num_of_cut = int(song.shape[0]/length_of_data)
        for c in range(num_of_cut):
            new_data.append(song[length_of_data*c:length_of_data*(c+1),:,:])
    new_data = np.array(new_data)
    new_data = new_data/new_data.max()
    print("cutting finished!")
    print("data size : ",new_data.shape)
    # data size :  (3199, 960, 128, 2)
    return new_data , label

#def main():
#    root_path = "raw_classical_music_data"
#    raw_data = load_file(root_path)
#    count = count_tab(raw_data)
#    cut_data , musician_label= make_data(raw_data)

########################################################
#                   operating function
########################################################
def make_mid(data, threshold):
    tarck1 = data[0][:,:,0]
    tarck2 = data[0][:,:,1]
    tarck1 = tarck1 >= threshold
    tarck2 = tarck2 >= threshold
    print(tarck1)
    print(tarck2)
    print("num of non-zero :",np.sum(tarck1)+np.sum(tarck2))
    tarck1 = Track(tarck1, program=0, is_drum=False, name='unknown')
    tarck2 = Track(tarck2, program=0, is_drum=False, name='unknown')
    return tarck1,tarck2

if __name__ == '__main__':
    test()
