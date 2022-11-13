# ====================
# 学習サイクルの実行
# ====================

# パッケージのインポート
from pathlib import Path
from single_network import single_network
from pv_descent import *
from train_network4 import train_network
from evaluate_network2 import *
from evaluate_best_player import *
import multiprocessing as mp
import sys
import torch
import random

def load_selfplay_data(select_num = 10):
    path_list = list(Path('./data').glob('*.history4'))
    if len(path_list) < select_num:
        return None
    return random.sample(path_list, select_num)

def clean_selfplay_data(select_num = 10):
    path_list = sorted(list(Path('./data').glob('*.history4')))
    if len(path_list) > select_num * 5:
        end = len(path_list) - (select_num * 5)
        for p in path_list[0:end]:
            p.unlink(True)

if __name__ == '__main__':

    mp.set_start_method('spawn')
    init_key()

    args = sys.argv
    self_play_num = 10
    epoch_num = 5
    batch_size = 512
    if len(args) >= 4:
        self_play_num = int(args[1])
        epoch_num = int(args[2])
        batch_size = int(args[3])

    print("GPU") if torch.cuda.is_available() else print("CPU")

    # デュアルネットワークの作成
    print(f"selfplay:{self_play_num}")
    print(f"epoch:{epoch_num}")
    print(f"batch:{batch_size}")
    i = 0
    while True:
        print('Train',i,'====================')
        load_data_list = load_selfplay_data(self_play_num)
        if load_data_list is None:
            print("not filll")
            time.sleep(60*15)
            continue
        # パラメータ更新部
        train_network(epoch_num, batch_size, load_data_list)
        # 新パラメータ評価部
        #evaluate_problem()
        update_best_player()
        if i % 10 == 0:
            evaluate_best_player()
        #evaluate_network()
        clean_selfplay_data(self_play_num)
        i += 1
