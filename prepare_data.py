import numpy as np
import torch
from torch.utils.data import Dataset
import random

def load_data(file_path):
    # file_path = '/home/ujjax/Documents/nips/pred-lstm/data/mnist_test_seq.npy'
    data = np.load(file_path)
    print('Data is of shape {}'.format(data.shape))
    return data


def get_batches(data, batch_size):
    length = len(data[0])
    for i in range(0, length, batch_size):
        batch = data[:, i:i+batch_size, :, :]  # [20,batch_size,64,64]
        batch_x = batch[0:10]
        batch_y = batch[10:20]
        yield (batch_x, batch_y)

'''
class BouncingMNistDataset(Dataset):

    It loads a batch of 200 at a time. When a batch of 200 is exhausted, it loads a new one.
    The order for loading files is random.
    For loading the data, initialized a non-shuffled dataloader.

    def __init__(self, root_dir, index_range=None, max_size=None):
        self.root_dir = root_dir
        if index_range is None:
            index_range = list(range(100))
        self.index_range = index_range
        if max_size is None:
            self.size = 800000
        else:
            self.size = max_size
        self.size_per_load = 200
        self.queue =[index_range[random.randint(0,len(index_range)-1)] for i in range(int(self.size/self.size_per_load))]
        self.cur_file_ind = 0
        self.data = None
    def __len__(self):
        return self.size
    def __getitem__(self, item):
        file_ind = self.queue[int(item/self.size_per_load)]
        if file_ind != self.cur_file_ind:
            self.data = np.load(self.root_dir+'/batch'+str(self.cur_file_ind)+'.npy')
            cur_file_ind = file_ind
        return self.data[:, item - file_ind * self.size_per_load,: , :, :]
'''