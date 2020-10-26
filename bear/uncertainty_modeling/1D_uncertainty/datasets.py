import numpy as np
from torch.utils.data import Dataset
import torch

class ScatterDataset(Dataset):
    def __init__(self, path):
        self.datas = np.load(path)
        self.X, self.Y = self.datas[:, 0], self.datas[:, 1]

    def __getitem__(self, index):
        x = np.array(self.X[index % len(self.X)])
        y = np.array(self.Y[index % len(self.Y)])
        w = torch.rand([64])

        return {'input' : x, 'label' : y, 'weight' : w}

    def __len__(self):
        return len(self.X)