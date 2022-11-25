# ====================
# 三目並べ
# ====================

# パッケージのインポート
import random
import math
import os
import pickle
from datetime import datetime

pos_dict = {}
ALL_POS_LEN = 5478

def append_pos_dict(k):
    if k in pos_dict:
        num = pos_dict[k]
        pos_dict[k] = num + 1
    else:
        pos_dict[k] = 1

def reset_pos_dict():
    pos_dict = {}

def len_pos_dict():
    return len(pos_dict)

# 学習データの保存
def write_data(history):
    now = datetime.now()
    os.makedirs('./base/', exist_ok=True) # フォルダがない時は生成
    path = './base/base.pos'
    with open(path, mode='wb') as f:
        pickle.dump(history, f)

# ゲーム状態
class State:
    # 初期化
    def __init__(self, pieces=None, enemy_pieces=None):
        # 石の配置
        self.pieces = pieces if pieces != None else [0] * 9
        self.enemy_pieces = enemy_pieces if enemy_pieces != None else [0] * 9

    # 石の数の取得
    def piece_count(self, pieces):
        count = 0
        for i in pieces:
            if i == 1:
                count +=  1
        return count

    # 全部の石の数
    def all_piece_count(self):
        return self.piece_count(self.pieces) + self.piece_count(self.enemy_pieces)

    # 負けかどうか
    def is_lose(self):
        # 3並びかどうか
        def is_comp(x, y, dx, dy):
            for k in range(3):
                if y < 0 or 2 < y or x < 0 or 2 < x or \
                    self.enemy_pieces[x+y*3] == 0:
                    return False
                x, y = x+dx, y+dy
            return True

        # 負けかどうか
        if is_comp(0, 0, 1, 1) or is_comp(0, 2, 1, -1):
            return True
        for i in range(3):
            if is_comp(0, i, 1, 0) or is_comp(i, 0, 0, 1):
                return True
        return False

    # 引き分けかどうか
    def is_draw(self):
        return self.piece_count(self.pieces) + self.piece_count(self.enemy_pieces) == 9

    # ゲーム終了かどうか
    def is_done(self):
        return self.is_lose() or self.is_draw()

    # 次の状態の取得
    def next(self, action):
        pieces = self.pieces.copy()
        pieces[action] = 1
        return State(self.enemy_pieces, pieces)

    # 合法手のリストの取得
    def legal_actions(self):
        actions = []
        for i in range(9):
            if self.pieces[i] == 0 and self.enemy_pieces[i] == 0:
                actions.append(i)
        return actions

    # 先手かどうか
    def is_first_player(self):
        return self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)

    # 45度回転
    def rotate45(self):
        dist = [6,3,0,7,4,1,8,5,2]
        tmp_pieces = [0]*9
        tmp_enemy_pieces = [0]*9
        for i, index in enumerate(dist):
            tmp_pieces[i] = self.pieces[index]
            tmp_enemy_pieces[i] = self.enemy_pieces[index]
        return State(tmp_pieces, tmp_enemy_pieces)
    # 左右反転
    def mirror(self):
        dist = [2,1,0,5,4,3,8,7,6]
        tmp_pieces = [0]*9
        tmp_enemy_pieces = [0]*9
        for i, index in enumerate(dist):
            tmp_pieces[i] = self.pieces[index]
            tmp_enemy_pieces[i] = self.enemy_pieces[index]
        return State(tmp_pieces, tmp_enemy_pieces)
    
    def hash_key(self):
        key = 0
        offset = 10**8
        ox = (2, 3) if self.is_first_player() else (3, 2)
        for piece, enemy in zip(self.pieces,self.enemy_pieces):
            tmp_key = 1
            if piece == 1:
                tmp_key = ox[0]
            elif enemy == 1:
                tmp_key = ox[1]
            key += tmp_key * offset
            offset /= 10
        return int(key)
    # 文字列表示
    def __str__(self):
        ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
        str = ''
        str = "turn:o\n" if self.is_first_player() else "turn:x\n"
        str += f"hash:{self.hash_key()}\n"
        for i in range(9):
            if self.pieces[i] == 1:
                str += ox[0]
            elif self.enemy_pieces[i] == 1:
                str += ox[1]
            else:
                str += '-'
            if i % 3 == 2:
                str += '\n'
        return str

# ランダムで行動選択
def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions)-1)]

# アルファベータ法で状態価値計算
def alpha_beta(state, alpha, beta):
    # 負けは状態価値-1
    if state.is_lose():
        return -1

    # 引き分けは状態価値0
    if state.is_draw():
        return  0

    # 合法手の状態価値の計算
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -beta, -alpha)
        if score > alpha:
            alpha = score

        # 現ノードのベストスコアが親ノードを超えたら探索終了
        if alpha >= beta:
            return alpha

    # 合法手の状態価値の最大値を返す
    return alpha

# アルファベータ法で行動選択
def alpha_beta_action(state):
    # 合法手の状態価値の計算
    best_action = 0
    alpha = -float('inf')
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -float('inf'), -alpha)
        if score > alpha:
            best_action = action
            alpha = score

    # 合法手の状態価値の最大値を持つ行動を返す
    return best_action

# プレイアウト
def playout(state):
    # 負けは状態価値-1
    if state.is_lose():
        return -1

    # 引き分けは状態価値0
    if state.is_draw():
        return  0

    # 次の状態の状態価値
    return -playout(state.next(random_action(state)))

# 最大値のインデックスを返す
def argmax(collection):
    return collection.index(max(collection))

# モンテカルロ木探索の行動選択
def mcts_action(state):
    # モンテカルロ木探索のノード
    class node:
        # 初期化
        def __init__(self, state):
            self.state = state # 状態
            self.w = 0 # 累計価値
            self.n = 0 # 試行回数
            self.child_nodes = None  # 子ノード群

        # 評価
        def evaluate(self):
            # ゲーム終了時
            if self.state.is_done():
                # 勝敗結果で価値を取得
                value = -1 if self.state.is_lose() else 0 # 負けは-1、引き分けは0

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

            # 子ノードが存在しない時
            if not self.child_nodes:
                # プレイアウトで価値を取得
                value = playout(self.state)

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1

                # 子ノードの展開
                if self.n == 10:
                    self.expand()
                return value

            # 子ノードが存在する時
            else:
                # UCB1が最大の子ノードの評価で価値を取得
                value = -self.next_child_node().evaluate()

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

        # 子ノードの展開
        def expand(self):
            legal_actions = self.state.legal_actions()
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(node(self.state.next(action)))

        # UCB1が最大の子ノードを取得
        def next_child_node(self):
             # 試行回数nが0の子ノードを返す
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node

            # UCB1の計算
            t = 0
            for c in self.child_nodes:
                t += c.n
            ucb1_values = []
            for child_node in self.child_nodes:
                ucb1_values.append(-child_node.w/child_node.n+2*(2*math.log(t)/child_node.n)**0.5)

            # UCB1が最大の子ノードを返す
            return self.child_nodes[argmax(ucb1_values)]

    # ルートノードの生成
    root_node = node(state)
    root_node.expand()

    # ルートノードを100回評価
    for _ in range(100):
        root_node.evaluate()

    # 試行回数の最大値を持つ行動を返す
    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]

def gen_pos_list():
    history = []
    # 状態の生成
    for i in range(200000):
        state = State()
        #print(f"\rtry:{i} num:{len_pos_dict()}",end="")
        #ゲーム終了までのループ
        while True:
            if state.hash_key() not in pos_dict:
                #history.append([state.pieces, state.enemy_pieces])
                history.append([0])
            append_pos_dict(state.hash_key())
            # ゲーム終了時
            if state.is_done():
                break

            # 次の状態の取得
            state = state.next(random_action(state))
    print(f"history_len:{len(history)}")
    #write_data(history)
# 動作確認
if __name__ == '__main__':
    gen_pos_list()