import torch
import h5py
from pathlib import Path
import numpy as np


class GW_SE_Dataset(object):

    def __init__(self):
        self.waveform_dataset = {'train': {'noisy': [], 'clean': []}, 'test': {'noisy': [], 'clean': []}}

    def save_waveform(self, DIR='.', data_fn='waveform_dataset.hdf5'):
        p = Path(DIR)
        p.mkdir(parents=True, exist_ok=True)

        f_data = h5py.File(p / data_fn, 'w')

        data_name = '0'
        for i in self.waveform_dataset.keys():
            for j in self.waveform_dataset[i].keys():
                data_name = i + '_' + j
                f_data.create_dataset(data_name, data=self.waveform_dataset[i][j], compression='gzip', compression_opts=9)
        f_data.close()

    def load_waveform(self, DIR='.', data_fn='waveform_dataset.hdf5'):
        p = Path(DIR)

        f_data = h5py.File(p / data_fn, 'r')
        data_name = '0'
        for i in self.waveform_dataset.keys():
            for j in self.waveform_dataset[i].keys():
                data_name = i + '_' + j
                self.waveform_dataset[i][j] = f_data[data_name][:, :]

        f_data.close()


class WaveformDatasetTorch(torch.utils.data.Dataset):
    def __init__(self, wfd, noise, train, length=4000):
        self.wfd = wfd
        self.noise = noise
        self.train = train
        self.type_str = 'train' if self.train else 'test'
        self.length = int(length)
        self.merge_dataset()
    
    def merge_dataset(self,):
        self.num = self.wfd.waveform_dataset[self.type_str]['clean'].shape[0] \
                    + self.noise.waveform_dataset[self.type_str]['clean'].shape[0]
        self.waveform_dataset = {   'train' :{'noisy':  np.zeros([self.num,self.length]),
                                              'clean':  np.zeros([self.num,self.length])},
                                    'test'  :{'noisy':  np.zeros([self.num,self.length]),
                                              'clean':  np.zeros([self.num,self.length])}}
        for i in self.waveform_dataset.keys():
            for j in self.waveform_dataset[i].keys():
                self.waveform_dataset[i][j]=np.vstack([self.wfd.waveform_dataset[i][j][:, -self.length:],
                                                       self.noise.waveform_dataset[i][j][:, -self.length:]])

    def __len__(self):
        return self.waveform_dataset[self.type_str]['clean'].shape[0] 
        
    def __getitem__(self, idx):
        noisy = self.waveform_dataset[self.type_str]['noisy'][idx][-self.length:]
        seq_len = np.array(noisy.shape[0])
        clean = self.waveform_dataset[self.type_str]['clean'][idx][-self.length:]
        label = 0 if np.allclose(clean,np.zeros(self.length)) else 1

        return torch.from_numpy(noisy).float(), \
                torch.from_numpy(seq_len), \
                torch.from_numpy(clean).float(),\
                torch.from_numpy(np.array([label]))

