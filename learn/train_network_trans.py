# ====================
# パラメータ更新部
# ====================

# パッケージのインポート
from generate_transformer_model import *
from generate_poolformer_model import *
from pathlib import Path
from history_dataset import *
from torch.utils.data import DataLoader
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import pandas as pd
import glob
import re
from evaluate_network2_trans import *
# パラメータの準備
RN_EPOCHS = 30 # 学習回数
RN_BATCH_SIZE = 128 # バッチサイズ
LAMBDA = 0.7

# ネットワークの学習
def train_network(epoch_num=RN_EPOCHS, batch_size=RN_BATCH_SIZE, path_list=None):

    iterate = 0

    if path_list is None:
        path_list = Path('./data').glob('*.json')

    # ベストプレイヤーのモデルの読み込み
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_list = glob.glob('model/*.save')
    #model = TransformerModel()
    model = PoolformerModel()
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.00001)

    if len(checkpoint_list) != 0:
        checkpoint_list.sort(key=lambda s: int(re.search(r'\d+', s).group()))
        #print(checkpoint_list)
        path = checkpoint_list[-1]
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        iterate = checkpoint['iterate'] + 1

    dataset = HistoryDataset(path_list,augmente=False)
    dataset_len = len(dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True) 
    start = time.time()
    sum_loss = 0
    sum_num = 0
    for i in range(epoch_num):
        print(f"epoch:{i}")
        sum_loss = 0
        sum_num = 0 
        for x, y0, y1 in dataloader:
            model.train()

            x = x.float().to(device)
            y0 = y0.float().to(device)
            y1 = y1.float().to(device)
                
            optimizer.zero_grad()
            outputs = model(x)
            outputs = torch.squeeze(outputs)
            
            # loss =  (LAMBDA * torch.sum((outputs - y0) ** 2)) + ((1 - LAMBDA) * torch.sum((outputs - y1) ** 2))
            loss =  torch.sum((outputs - y0) ** 2)

            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            sum_num += 1
            if sum_num % 1000 == 0:
                n = batch_size * sum_num
                now = time.time()
                print(f"{n}/{dataset_len} ({100 * (n/dataset_len):.3f}%) loss:{sum_loss/sum_num} sec:{int(now-start)}")

        print(f"avg loss:{sum_loss / sum_num}")
    
    # 最新プレイヤーのモデルの保存
    torch.save(model.state_dict(), './model/best_single.h5')
    save_checkpoint(iterate=iterate, model=model, optimizer=optimizer)
    conv_jit()
    ans = evaluate_problem()
    dump_loss(sum_loss/sum_num,ans)

def dump_loss(loss,ans):
    selfplay_df = pd.read_csv("selfplay_result.csv")
    selfplay_df.iloc[-1, selfplay_df.columns.get_loc("loss")] = loss  
    selfplay_df.iloc[-1, selfplay_df.columns.get_loc("ans")] = ans  
    if os.path.isfile("history.csv"):
        history_df = pd.read_csv("history.csv")
        history_df = pd.concat([history_df, selfplay_df],axis=0)
    else:
        history_df = selfplay_df.copy()
    history_df.to_csv("history.csv",index=False)

def save_checkpoint(iterate, model, optimizer):
    path = f"model/iterate{iterate}.save"
    checkpoint = {
        'iterate': iterate,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)

# 動作確認
if __name__ == '__main__':
    train_network()
