# ====================
# 新パラメータ評価部
# ====================

# パッケージのインポート
from game import State
from pathlib import Path
from shutil import copy
import numpy as np
from generate_transformer_model import *
from generate_poolformer_model import *
import json

# ベストプレイヤーの交代
def update_best_player():
    copy('./model/latest_single.h5', './model/best_single.h5')
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
    #model = TransformerModel()
    model = PoolformerModel()
    model.load_state_dict(torch.load('./model/best_single.h5'))
    model = model.to(device)
    model.eval()

    # 状態の生成

    with open('../oracle/oracle_result.json') as f:
        problem_list = json.load(f)
    
    correct_num = 0
    for problem in problem_list:
        key = problem["p"]
        result = problem["r"]
        state = from_hash(key)
        score = predict(model,state,device)
        score = score.item()
        org_score = score
        if score > 0.2:
            score = 1
        elif score < -0.2:
            score = -1
        else:
            score = 0
        if state.is_lose():
            score = -1
        elif state.is_draw():
            score = 0
        elif state.is_win():
            score = 1

        if score == result:
            correct_num += 1
        # else:
        #     print(state)
        #     print("org:",org_score)
        #     print("score:",score)
        #     print("result:",result)
    print("ans:",correct_num/len(problem_list),f"({correct_num})/({len(problem_list)})")
    return correct_num/len(problem_list)    
    
# 動作確認
if __name__ == '__main__':
    evaluate_problem()
