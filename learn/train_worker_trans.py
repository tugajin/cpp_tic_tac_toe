# ====================
# 学習サイクルの実行
# ====================

# パッケージのインポート
from pathlib import Path
from train_network_trans import train_network
from evaluate_network_trans import *
import multiprocessing as mp
import sys
import torch
import random
import time

def load_selfplay_data():
    path_list = list(Path('./data').glob('*.json'))
    return path_list

def clean_selfplay_data():
    path_list = list(Path('./data').glob('*.json'))
    for p in path_list:
        p.unlink(True)

if __name__ == '__main__':

    mp.set_start_method('spawn')

    args = sys.argv
    epoch_num = 30 
    batch_size = 128

    if len(args) >= 3:
        epoch_num = int(args[1])
        batch_size = int(args[2])

    print("GPU") if torch.cuda.is_available() else print("CPU")

    print(f"epoch:{epoch_num}")
    print(f"batch:{batch_size}")
        
    load_data_list = load_selfplay_data()
    # パラメータ更新部
    train_network(epoch_num, batch_size, load_data_list)
    # 新パラメータ評価部
    update_best_player()
    conv_jit()
    clean_selfplay_data()
