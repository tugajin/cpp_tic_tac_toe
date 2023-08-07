# ====================
# 新パラメータ評価部
# ====================

# パッケージのインポート
from game import State
from pathlib import Path
from shutil import copy
import numpy as np
from single_network import *
import datetime

# ベストプレイヤーの交代
def update_best_player():
    dt_now = datetime.datetime.now()
    copy('./model/latest_single.h5', './model/best_single.h5')
    copy('./model/latest_single.h5', f'./model/best_single_{dt_now.isoformat()}.h5')
    print('Change BestPlayer')

def predict(model, state, device):
    feature = np.array(state.feature())
    file, rank, channel = DN_INPUT_SHAPE
    feature = feature.reshape(1, channel, file, rank)
    f = torch.from_numpy(feature)
    f = f.to(device).float()
    return model(f)

def evaluate_problem():
    # ベストプレイヤーのモデルの読み込み
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    model = torch.jit.load('./model/best_single_jit.pt')
    model = model.to(device)

    # 状態の生成
    state = State()
    print(state)
    print(predict(model,state,device))
    print("---------------------")
    
    
# 動作確認
if __name__ == '__main__':
    #evaluate_network()
    evaluate_problem()
