from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import pickle
from single_network import *
import numpy as np
import time

# class HistoryDataset(Dataset):
#     def __init__(self, root):
#         super().__init__()
#         self.data = []
#         for path in root:
#             with path.open(mode='rb') as f:
#                 self.data.extend(pickle.load(f))
#         self.data = self.data[0:100]
#     # ここで取り出すデータを指定している
#     def __getitem__(self, index) :
#         data = np.array(self.data[index][0])
#         file, rank, channel = DN_INPUT_SHAPE
#         data = data.reshape(channel, file, rank)
#         y_deep = self.data[index][1]
#         c = self.data[index][2]
#         r = self.data[index][3]

#         return data, y_deep, c, r

#     # この method がないと DataLoader を呼び出す際にエラーを吐かれる
#     def __len__(self):
#         return len(self.data)

class HistoryDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        data = []
        for path in root:
            with path.open(mode='rb') as f:
                data.extend(pickle.load(f))
        data2 = []
        for d in data:
            dd = np.array(d[0])
            file, rank, channel = DN_INPUT_SHAPE
            dd = dd.reshape(channel, file, rank)
            data2.append([dd, d[1],d[2],d[3]])
        self.data = data2
    # ここで取り出すデータを指定している
    def __getitem__(self, index) :
        data = self.data[index][0]
        y_deep = self.data[index][1]
        c = self.data[index][2]
        r = self.data[index][3]
        return data, y_deep, c, r

    # この method がないと DataLoader を呼び出す際にエラーを吐かれる
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    history_path = sorted(Path('./data').glob('*.history4'))[-1]
    dataset = HistoryDataset([history_path]) 
 
