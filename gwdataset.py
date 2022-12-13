import pycbc.psd
import numpy as np
from time import gmtime, time, strftime, localtime
# import gwsurrogate
import os
import h5py
import functools
from pathlib import Path
import itertools
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

EPS = 1e-5


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





def main():
    return 0


if __name__ == '__main__':
    main()