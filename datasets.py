from sys import path
import h5py
import scipy.io
import math
import pandas as pd
from tensorflow.keras.utils import Sequence
import numpy as np
import utils

class ECGSequence(Sequence):
    @classmethod
    def get_train_and_val(cls, path_to_data, dset, path_to_annotations, batch_size=8, val_split=0.02):
        n_samples = utils.get_file_num(path_to_annotations, '.hea')
        n_train = math.ceil(n_samples*(1-val_split))
        train_seq = cls(path_to_data, dset, path_to_annotations, batch_size, end_idx=n_train)
        valid_seq = cls(path_to_data, dset, path_to_annotations, batch_size, start_idx=n_train)
        return train_seq, valid_seq

    def __init__(self, path_to_data, hdf5_dset, path_to_annotations=None, batch_size=8,
                 start_idx=0, end_idx=None):
        if path_to_annotations is None:
            self.y = None
        else:
            self.y = np.array(utils.get_all_hea(path_to_annotations))
        # Get tracings
        self.x = np.array(utils.get_all_mat(path_to_data, hdf5_dset))
        self.batch_size = batch_size
        if end_idx is None:
            end_idx = len(self.x)
        self.start_idx = start_idx
        self.end_idx = end_idx

    @property
    def n_classes(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        if self.y is None:
            return np.array(self.x[start:end, :, :])
        else:
            return np.array(self.x[start:end, :, :]), np.array(self.y[start:end])

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

if __name__ == "__main__":
    y = np.array(utils.get_all_hea("./newData/"))
    x = np.array(utils.get_all_mat("./newData/", 'val'))
    print(x[0].shape)