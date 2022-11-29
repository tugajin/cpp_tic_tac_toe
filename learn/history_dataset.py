from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import json
from single_network import *
import numpy as np
import time
from game import *

class HistoryDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        data = []
        for path in root:
            with path.open(mode='r') as f:
                try:
                    d = json.loads(f.read())
                except json.decoder.JSONDecodeError:
                    print(f"decode error:{path}")
                    pass
                data.extend(d)
        data2 = []
        for d in data:
            state = from_hash(d["p"])
            score = d["s"]
            file, rank, channel = DN_INPUT_SHAPE
            feature = np.array(state.feature())
            feature = feature.reshape(channel, file, rank)
            data2.append([feature, score])
        self.data = data2
        print(f"len:{len(data)}")
    # ここで取り出すデータを指定している
    def __getitem__(self, index) :
        data = self.data[index][0]
        y_deep = self.data[index][1]
        return data, y_deep

    # この method がないと DataLoader を呼び出す際にエラーを吐かれる
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    history_path = sorted(Path('./data').glob('*.json'))
    dataset = HistoryDataset(history_path) 
 
