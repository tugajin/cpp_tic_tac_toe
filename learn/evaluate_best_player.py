# ====================
# ベストプレイヤーの評価
# ====================

# パッケージのインポート
from game import State, random_action, alpha_beta_action, mcts_action
from pv_mcts import pv_mcts_action
from pathlib import Path
import numpy as np
from dual_network import *

# パラメータの準備
EP_GAME_COUNT = 10  # 1評価あたりのゲーム数

# 先手プレイヤーのポイント
def first_player_point(ended_state):
    # 1:先手勝利, 0:先手敗北, 0.5:引き分け
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 0.5

# 1ゲームの実行
def play(next_actions):
    # 状態の生成
    state = State()

    # ゲーム終了までループ
    while True:
        # ゲーム終了時
        if state.is_done():
            break

        # 行動の取得
        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        # 次の状態の取得
        state = state.next(action)

    # 先手プレイヤーのポイントを返す
    return first_player_point(state)

# 任意のアルゴリズムの評価
def evaluate_algorithm_of(label, next_actions):
    # 複数回の対戦を繰り返す
    total_point = 0
    for i in range(EP_GAME_COUNT):
        # 1ゲームの実行
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(list(reversed(next_actions)))

        # 出力
        print('\rEvaluate {}/{}'.format(i + 1, EP_GAME_COUNT), end='')
    print('')

    # 平均ポイントの計算
    average_point = total_point / EP_GAME_COUNT
    print(label, average_point)

# ベストプレイヤーの評価
def evaluate_best_player():
    # ベストプレイヤーのモデルの読み込み
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualNet()
    model.load_state_dict(torch.load("./model/best.h5"))
    model = model.to(device)
    # PV MCTSで行動選択を行う関数の生成
    next_pv_mcts_action = pv_mcts_action(model, 0.0)

    # VSランダム
    next_actions = (next_pv_mcts_action, random_action)
    evaluate_algorithm_of('VS_Random', next_actions)

    # VSアルファベータ法
    next_actions = (next_pv_mcts_action, alpha_beta_action)
    evaluate_algorithm_of('VS_AlphaBeta', next_actions)

    # VSモンテカルロ木探索
    next_actions = (next_pv_mcts_action, mcts_action)
    evaluate_algorithm_of('VS_MCTS', next_actions)

# 動作確認
if __name__ == '__main__':
    evaluate_best_player()
