from torch.utils import data
import numpy as np
import pandas as pd


class Dataset(data.Dataset):

    def __init__(self, file_path):
        self.X = pd.read_csv(file_path).values[:, 1:]/255
        self.x_dim = np.shape(self.X)[1]

        self.L = pd.read_csv(file_path).values[:, 0]
        self.num_label = len(np.unique(self.L))
        self.num_sample = len(self.L)

        self.Y = np.zeros((len(self.L), self.num_label))
        self.Y[np.arange(self.num_sample), self.L] = 1

    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        ll = self.L[index]
        return x, y, ll
