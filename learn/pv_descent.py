# ====================
# Descentの作成
# ====================

# パッケージのインポート
from math import sqrt
from multiprocessing import Process
from game import *
from pv_ubfm import pv_ubfm_scores
from datetime import datetime
from pathlib import Path
from single_network import *
from db import *
from mate_search import *

import tracemalloc
import random
import operator
import numpy as np
import pickle
import os
import torch
import time
import multiprocessing as mp

# パラメータの準備
PV_EVALUATE_COUNT = 50 # 1推論あたりのシミュレーション回数（本家は1600）
# パラメータの準備
SP_GAME_COUNT = 10 # セルフプレイを行うゲーム数（本家は25000）
SP_TEMPERATURE = 1.0 # ボルツマン分布の温度パラメータ

# 推論
def predict(model, node_list, device):

    #print("predict")
    #print(state)
    # 推論のための入力データのシェイプの変換
    file, rank, channel = DN_INPUT_SHAPE
    x = np.array([node.state.pieces_array() for node in node_list])

    #print(x)

    x = x.reshape(len(node_list), channel, file, rank)
    x = np.array(x)
    x = torch.tensor(x,dtype=torch.float32)
   
    x = x.to(device)
    
    with torch.no_grad():
        # 推論
        y = model(x)
    # 価値の取得
    for i in range(len(node_list)):
        value = y[i][0].item()
        # 丸め
        value = (int(value * 10000))/10000
        if value >= 1:
            value = 0.9000
        elif value <= -1:
            value = -0.9000
        node_list[i].w = value
        node_list[i].n += 1
        if node_list[i].state.is_done():
            node_list[i].w = score_lose(node_list[i].ply) if node_list[i].state.is_lose() else 0
            node_list[i].completion = -1 if node_list[i].state.is_lose() else 0
            node_list[i].resolved = True
        elif node_list[i].state.is_win():
            node_list[i].w = score_win(node_list[i].ply+1) 
            node_list[i].completion = 1
            node_list[i].resolved = True
        #elif mate_action(node_list[i].state) is not None:
            #print("------------------------------")
            #print("found mate")
            #print(node_list[i].state)
            #print("------------------------------")
            # FIXME 何手で詰んだかわからないので、適当な値
            # resolvedにしてしまうと詰み局面を学習対象にできない
         #   node_list[i].w = score_win(node_list[i].ply+5)
            #node_list[i].completion = 1
            #node_list[i].resolved = True
        #elif mated_action(node_list[i].state) is None:
            #print("--------------------------------")
            #print("found mated")
            #print(node_list[i].state)
            #print("--------------------------------")
        #    node_list[i].w = score_lose(node_list[i].ply+4)

# ノードのリストを試行回数のリストに変換
def nodes_to_scores(nodes):
    scores = [0] * len(nodes)
    for i, c in enumerate(nodes):
        if c.resolved:
            if c.completion == -1:
                p = 0.99
            elif c.completion == 0:
                p = 0.5
            elif c.completion == 1:
                p = 0.01
            else:
                assert(False)
            scores[i] = p
        else:
            scores[i] = 1.0 - c.w
    return scores

def score_win(ply):
    return 0.9999 - (ply / 100) 
def score_lose(ply):
    return -score_win(ply)

# Descent木探索のスコアの取得
def pv_descent_scores(model, state, device, history, temperature):

    # モンテカルロ木探索のノードの定義
    class Node:
        # ノードの初期化
        def __init__(self, state, ply, action = -1):
            self.state = state # 状態
            self.w = -999 # 価値
            self.n = 0 # 試行回数
            self.ply = ply
            self.child_nodes = None  # 子ノード群
            self.action = action
            self.best_action = -1
            self.resolved = False
            self.completion = -99 # draw:0 win:1 lose:-1 unknown: -99

        def dump(self, only_root = False):
            print("-----------------start-------------------------")
            print(self.state)
            print("w:",self.w)
            print("n:",self.n)
            print("ply:",self.ply)
            print("action:",self.state.action_str(self.action))
            print("best_action",self.state.action_str(self.best_action))
            print("has_child:",not self.child_nodes is None)
            print("resoloved:",self.resolved)
            print("completion:",self.completion)
            if self.child_nodes:
                for c in self.child_nodes:
                    print("  action:",self.state.action_str(c.action))
                    print("  child_w:",c.w)
                    print("  child_n:",c.n)
                    print("  child_ply:",c.ply)
                    print("  resoloved:",c.resolved)
                    print("  completion:",c.completion)
                    print("  ---------------------")
                    
            print("-----------------end-------------------------")
        # 局面の価値の計算
        def evaluate(self):
            assert(not self.resolved)
            assert(not self.state.is_done())
            # # ゲーム終了時
            # if self.state.is_done():
            #     # 勝敗結果で価値を取得
            #     self.w = self.completion = -1 if self.state.is_lose() else 0
            #     # 試行回数の更新
            #     self.n += 1
            #     self.resolved = True
            #     assert(False)
            #     return 

            # 子ノードが存在しない時
            if not self.child_nodes:
                # 子ノードの展開
                self.child_nodes = []
                for action in self.state.legal_actions():
                    self.child_nodes.append(Node(self.state.next(action),self.ply+1,action))
                # ニューラルネットワークの推論で価値を取得
                predict(model, self.child_nodes, device)
                # 価値と試行回数の更新
                self.update_node()
            if not self.resolved:
                # 評価値が最大の子ノードを取得
                next_node = self.next_child_node()
                assert(next_node is not None)
                assert(not next_node.resolved)
                assert(not next_node.state.is_done())
                next_node.evaluate()
                # 価値と試行回数の更新
                self.update_node()

        def next_child_node_debug(self):
            max_index = -1
            max_value = -9999
            min_num = self.n
            lose_num = 0
            draw_num = 0
            child_nodes_len = len(self.child_nodes)
            assert(child_nodes_len != 0)
            for i, child_node in enumerate(self.child_nodes):
                if child_node.resolved:
                    # 子供に負けを見つけた→つまり勝ちなので終わり
                    if child_node.completion == -1:
                        max_index = i
                        self.resolved = True
                        self.completion = 1
                        self.w = 1
                        self.dump(True)
                        assert(False)
                        return None
                    elif child_node.completion == 1:
                        lose_num += 1
                    else:
                        assert(child_node.completion == 0)
                        draw_num += 1
                else:
                    if -child_node.w == max_value:
                        if child_node.n < min_num:
                            max_index = i
                            max_value = -child_node.w
                            min_num = child_node.n
                    elif -child_node.w > max_value:
                        max_index = i
                        max_value = -child_node.w
                        min_num = child_node.n
            #子供が全部引き分け
            if child_nodes_len == draw_num:
                self.resolved = True
                self.completion = 0
                self.w = 0
                self.dump(True)
                assert(False)
                return None
            # 子供が全部勝ち→この局面は負け
            elif child_nodes_len == lose_num:
                self.resolved = True
                self.completion = -1
                self.w = -1
                self.dump(True)
                assert(False)
                return None
            # 子供に引き分けと負けが付与済→引き分け
            elif child_nodes_len == (draw_num + lose_num):
                assert(draw_num != 0)
                self.resolved = True
                self.completion = 0
                self.w = 0
                self.dump(True)
                assert(False)
                return None
            assert(not self.child_nodes[max_index].resolved)
            return self.child_nodes[max_index]
        
        # 評価値が最大の子ノードを取得
        def next_child_node(self):
            not_resolved = [child for child in self.child_nodes if not child.resolved]
            assert(len(not_resolved) != 0)
            max_child = max(not_resolved,key=lambda x: (-x.w, -x.n) )
            # debug_max_child = self.next_child_node_debug()
            # if max_child.action != debug_max_child.action:
            #     self.dump(True)
            #     print(max_child.action)
            #     print(debug_max_child.action)
            #     assert(False)
            return max_child

        # 現在のnodeの情報を更新
        def update_node(self):
            max_index = -1
            max_value = -9999
            max_num = -1
            lose_num = 0
            draw_num = 0
            child_nodes_len = len(self.child_nodes)
            assert(child_nodes_len != 0)
            for i, child_node in enumerate(self.child_nodes):
                if child_node.resolved:
                    # 子供に負けを見つけた→つまり勝ちなので終わり
                    if child_node.completion == -1:
                        max_index = i
                        self.resolved = True
                        self.completion = 1
                        self.w = -child_node.w
                        self.best_action = child_node.action
                        return 
                    elif child_node.completion == 1:
                        lose_num += 1
                        #負けの局面は選ばない
                        #continue
                    else:
                        assert(child_node.completion == 0)
                        draw_num += 1
                if -child_node.w == max_value:
                    if child_node.n > max_num:
                        max_index = i
                        max_value = -child_node.w
                        max_num = child_node.n
                elif -child_node.w > max_value:
                    max_index = i
                    max_value = -child_node.w
                    max_num = child_node.n
            self.n += 1
            #子供が全部引き分け
            if child_nodes_len == draw_num:
                self.resolved = True
                self.w = self.completion = 0
                return 
            # 子供が全部勝ち→この局面は負け
            elif child_nodes_len == lose_num:
                self.resolved = True
                self.completion = -1
                self.w = -self.child_nodes[max_index].w
                self.best_action = self.child_nodes[max_index].action
                return 
            # 子供に引き分けと負けが付与済→引き分け
            elif child_nodes_len == (draw_num + lose_num):
                assert(draw_num != 0)
                self.resolved = True
                self.w = self.completion = 0
                return 
            self.w = -self.child_nodes[max_index].w
            self.best_action = self.child_nodes[max_index].action

    # 現在の局面のノードの作成
    root_node = Node(state, 0)

    # 複数回の評価の実行
    #print("start simulation")
    for i in range(PV_EVALUATE_COUNT):
        #print(f"try:{i} \r",end="")
        if root_node.resolved:
            break
        root_node.evaluate()
    #    root_node.dump()
    #    print(f"result:{result}")
    #print("end simulation")

    # 合法手の確率分布
    scores = nodes_to_scores(root_node.child_nodes)

    # 探索木の情報をhistoryに登録
    add_history(history, root_node)

    n = root_node
   # n.dump()

    #while True:
    while False:
        n.dump()
        if not n.child_nodes:
            break
        best_child = n.next_child_node()
        if best_child is None:
            break
        n = best_child

    if temperature == 0: # 最大値のみ1
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else: # ボルツマン分布でバラつき付加
        scores = boltzman(scores, temperature)
    return scores, root_node.w

# ボルツマン分布
def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]

# 探索木の情報を保存
def add_history(history, node):
    s = node.state
    w = node.w
    c = node.completion
    r = node.resolved
    if mate_action(s) is not None:
        w = score_win(node.ply+5)
        c = 1
        r = True
    elif mated_action(s) is None:
        w = score_lose(node.ply+4)
        c = -1
        r = True
    history.append([node.state.pieces_array(), w, c, r])
    # このノードの情報を格納
    if node.child_nodes is not None:
        for child in node.child_nodes:
            add_history(history,child)

# 学習データの保存
def write_data(history):
    now = datetime.now()
    os.makedirs('./data/', exist_ok=True) # フォルダがない時は生成
    path = './data/{:04}{:02}{:02}{:02}{:02}{:02}{:02}.history4'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond)
    with open(path, mode='wb') as f:
        pickle.dump(history, f)

# ε-greedyで選ぶ
def e_greedy(l, scores, p):
    # 学習データに状態と方策を追加
    if random.random() < p:
        action = np.random.choice(l)
    else:
        # 行動の取得
        action = l[np.argmax(scores)]
    return action
def ordinal(l, scores, p):
    i = 0
    while True:
        index = np.argmax(scores)
        if random.random() < p:
            #print(f"\n{i}")
            return l[index]
        scores[index] = -1
        i+=1
def ab(state):
    return alpha_beta_action4(state)
# 1ゲームの実行
def play(model, device):
    # 学習データ
    history = []

    # 状態の生成
    state = State()
    i = 0
    result = 0
    value_history = []
    while True:
        print(f"{i} ",end="\r")
        #print(state)
        # ゲーム終了時
        if state.is_done():
            result = 1 if state.is_lose() else 0 
            break
        
        # 合法手の確率分布の取得
        scores, values = pv_descent_scores(model, state, device, history, SP_TEMPERATURE)
        value_history.append(values)
        #action = e_greedy(state.legal_actions(), scores, 0.03)
        action = ordinal(state.legal_actions(), scores, 0.8)
        # 次の状態の取得
        state = state.next(action)
        i += 1
        
    return history, result

# セルフプレイ
def self_play(self_play_num = SP_GAME_COUNT):
    single_network()
    # ベストプレイヤーのモデルの読み込み
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    model = SingleNet()
    model.load_state_dict(torch.load('./model/best_single.h5',device))
    model = model.to(device)
    model.eval()

    # 複数回のゲームの実行
    for i in range(self_play_num):
        if i % 5 == 0:
            print("load model")
            model = SingleNet()
            model.load_state_dict(torch.load('./model/best_single.h5',device))
            model = model.to(device)
            model.eval()

        # 1ゲームの実行
        h, r = play(model, device)
        # 出力
        print('\rSelfPlay {}/{} {}'.format(i+1, self_play_num,r), end='')
        # 学習データの保存
        write_data(h)
# 動作確認
if __name__ == '__main__':
    init_key()
    self_play(100000000000)
