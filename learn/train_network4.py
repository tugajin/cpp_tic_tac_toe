# ====================
# パラメータ更新部
# ====================

# パッケージのインポート
from single_network import *
from pathlib import Path
from history_dataset import *
from torch.utils.data import DataLoader
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import random

# パラメータの準備
RN_EPOCHS = 20 # 学習回数
RN_BATCH_SIZE = 128 # バッチサイズ

# 学習データの読み込み
def load_data():
    history_path = sorted(Path('./data').glob('*.history4'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)
# 学習データの読み込み
def load_all_data():
    history_path = sorted(Path('./data').glob('*.history4'))
    ret = []
    for path in history_path:
        with path.open(mode='rb') as f:
            tmp = pickle.load(f)
            ret.extend(tmp)
    return ret
# ネットワークの学習
def train_network():
    
    # ベストプレイヤーのモデルの読み込み
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SingleNet()
    model.load_state_dict(torch.load('./model/best_single.h5'))
    model = model.to(device)
    
    model.train()
    optimizer = optim.SGD(model.parameters(),lr=0.001, weight_decay=0.00001)
    dataset = HistoryDataset([sorted(Path('./data').glob('*.history4'))[-1]])
    dataset_len = len(dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=RN_BATCH_SIZE, shuffle=True) 
    for i in range(0,RN_EPOCHS):
        print(f"epoch:{i}")
        sum_loss = 0
        sum_num = 0 
        for x, y, c, r in dataloader:
            x = x.float().to(device)
            y = y.float().to(device)
                
            optimizer.zero_grad()
            outputs = model(x)
            outputs = torch.squeeze(outputs)
            loss = torch.sum((outputs - y) ** 2)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            sum_num += 1
            if sum_num % 1000 == 0:
                n = RN_BATCH_SIZE * sum_num
                print(f"{n}/{dataset_len} ({100 * (n/dataset_len):.3f}%) loss:{loss.item()}")

        print(f"avg loss:{sum_loss / sum_num}")

    # 最新プレイヤーのモデルの保存
    torch.save(model.state_dict(), './model/latest_single.h5')

def check_train_data():
    # 学習データの読み込み
    history = load_all_data()
    print(type(history))
    for hist in history:
        x, y_deep, c, r = zip(hist)
        state = State(x[0][0],x[0][1])
        print(state)
        print(f"deep:{y_deep}")
        print(f"c:{c}")
        print(f"r:{r}")
        print("------------------------")

# 動作確認
if __name__ == '__main__':
    train_network()
    #check_train_data()