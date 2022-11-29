# ====================
# 新パラメータ評価部
# ====================

# パッケージのインポート
from game import State
from pathlib import Path
from shutil import copy
import numpy as np
from single_network import *

# ベストプレイヤーの交代
def update_best_player():
    copy('./model/latest_single.h5', './model/best_single.h5')
    print('Change BestPlayer')

def predict(model,model2, state, device):
    feature = np.array(state.feature())
    file, rank, channel = DN_INPUT_SHAPE
    feature = feature.reshape(1, channel, file, rank)
    f = torch.from_numpy(feature)
    f = f.to(device).float()
    return model(f),model2(f)

def evaluate_problem():
    # ベストプレイヤーのモデルの読み込み
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    model = SingleNet()
    model.load_state_dict(torch.load('./model/best_single.h5'))
    model = model.to(device)
    model.eval()

    model2 = torch.jit.load('./model/best_single_jit.pt')
    model2 = model.to(device)

    # 状態の生成
    state = State()
    print(state)
    print(predict(model,model2,state,device))
    print("---------------------")
    
    state = state.next(val_to_move(2))
    print(state)
    print(predict(model,model2,state,device))

    print("---------------------")

    state = state.next(val_to_move(1))
    print(state)
    print(predict(model,model2,state,device))
    print("---------------------")

    state = state.next(val_to_move(4))
    print(state)
    print(predict(model,model2,state,device))
    print("---------------------")

    state = state.next(val_to_move(6))
    print(state)
    print(predict(model,model2,state,device))
    print("---------------------")

    state = State()
    state = state.next(val_to_move(2)) 
    state = state.next(val_to_move(0)) 
    state = state.next(val_to_move(4)) 
    state = state.next(val_to_move(1)) 
    print(state)
    print(predict(model,model2,state,device))
    print("---------------------")

    state = State()
    state = state.next(val_to_move(2)) 
    state = state.next(val_to_move(0))
    state = state.next(val_to_move(5)) 
    state = state.next(val_to_move(1)) 
    print(state)
    print(predict(model,model2,state,device))
    print("---------------------")

    state = State()
    state = state.next(val_to_move(0)) 
    state = state.next(val_to_move(6)) 
    state = state.next(val_to_move(1)) 
    state = state.next(val_to_move(7))
    print(state)
    print(predict(model,model2,state,device))

    state = State()
    state = state.next(val_to_move(2)) 
    state = state.next(val_to_move(7)) 
    print(state)
    print(predict(model,model2,state,device))
    
# 動作確認
if __name__ == '__main__':
    #evaluate_network()
    evaluate_problem()
