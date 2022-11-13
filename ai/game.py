# ====================
# 簡易将棋
# ====================

# パッケージのインポート
import random
import math
import random

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

self_piece_key = [[]]
self_hand_piece_key = [[]]
enemy_piece_key = [[]]
enemy_hand_piece_key = [[]]
turn_key = []
class HashKey():
    def __init__(self):
        self.x = 123456789
        self.y = 362436069
        self.z = 521288629
        self.w = 88675123
    def to32(self, x):
        return (x & 4294967295)
    def get(self):
        t = self.to32((self.x ^ self.to32((self.x<<11))))
        self.x = self.y
        self.y = self.z
        self.z = self.w
        self.w = (self.w ^ self.to32((self.w>>19))) ^ (t ^ self.to32((t>>8)))
        return self.w
    
def init_key():
    h = HashKey()
    global self_piece_key
    global enemy_piece_key
    global self_hand_piece_key
    global enemy_hand_piece_key
    global turn_key

    self_piece_key  = [[h.get() for i in range(12)] for j in range(5)]
    enemy_piece_key = [[h.get() for i in range(12)] for j in range(5)]
    self_hand_piece_key = [ [ h.get() for i in range(3) ] for j in range(5) ]
    enemy_hand_piece_key = [ [ h.get() for i in range(3) ] for j in range(5) ]
    turn_key = [h.get() for i in range(2)]

# ゲームの状態
class State:
    # 初期化
    def __init__(self, pieces=None, enemy_pieces=None, history=[]):
        # 方向定数
        self.dxy = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))

        # 駒の配置
        self.pieces = pieces if pieces != None else [0] * (12+3)
        self.enemy_pieces = enemy_pieces if enemy_pieces != None else [0] * (12+3)
        self.history = history

        # 駒の初期配置
        if pieces == None or enemy_pieces == None:
            self.pieces       = [0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 4, 3, 0, 0, 0]
            self.enemy_pieces = [0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 4, 3, 0, 0, 0]

    # 負けかどうか
    def is_lose(self):
        for i in range(12):
            if self.pieces[i] == 4: # ライオン存在
                return len(self.legal_actions()) == 0
        return True

    # ライオンが取れるか？
    def is_win(self):
        actions = self.legal_actions()
        for action in actions:
            position_dst, _ = self.action_to_position(action)
            if self.enemy_pieces[11-position_dst] == 4:
                return True
        return False
    
    # 王手されているか？
    def in_checked(self):
        tmp_pieces = self.pieces.copy()
        tmp_enemy_pieces = self.enemy_pieces.copy()
        self.pieces = tmp_enemy_pieces.copy()
        self.enemy_pieces = tmp_pieces.copy()
        ret = self.is_win()
        self.pieces = tmp_pieces.copy()
        self.enemy_pieces = tmp_enemy_pieces.copy()
        return ret

    # 引き分けかどうか
    def is_draw(self):
        current_key = self.hash_key()
        i = len(self.history) - 4
        while i >= 0:
            if self.history[i] == current_key:
                return True
            i -= 4
        return len(self.history) >= 128

    # ゲーム終了かどうか
    def is_done(self):
        return self.is_lose() or self.is_draw()

    # デュアルネットワークの入力の2次元配列の取得
    def pieces_array(self):
        # プレイヤー毎のデュアルネットワークの入力の2次元配列の取得
        def pieces_array_of(pieces):
            table_list = []
            # 0:ヒヨコ, 1:ゾウ, 2:キリン, 3:ライオン,
            for j in range(1, 5):
                table = [0] * 12
                table_list.append(table)
                for i in range(12):
                    if pieces[i] == j:
                        table[i] = 1

            # 4:ヒヨコの持ち駒, 5:ゾウの持ち駒, 6:キリンの持ち駒
            for j in range(1, 4):
                flag = 1 if pieces[11+j] > 0 else 0
                table = [flag] * 12
                table_list.append(table)
            return table_list

        # デュアルネットワークの入力の2次元配列の取得
        return [pieces_array_of(self.pieces), pieces_array_of(self.enemy_pieces)]

    # 駒の移動先と移動元を行動に変換
    def position_to_action(self, position, direction):
        return position * 11 + direction

    # 行動を駒の移動先と移動元に変換
    def action_to_position(self, action):
        return (int(action/11), action%11)

    # 合法手のリストの取得
    def legal_actions(self):
        actions = []
        for p in range(12):
            # 駒の移動時
            if self.pieces[p]  != 0:
                actions.extend(self.legal_actions_pos(p))

            # 持ち駒の配置時
            if self.pieces[p] == 0 and self.enemy_pieces[11-p] == 0:
                for capture in range(1, 4):
                    if self.pieces[11+capture] != 0:
                        actions.append(self.position_to_action(p, 8-1+capture))
        return actions

    # らいおんが取られる手を生成しない
    def perfect_legal_actions(self):
        actions = self.legal_actions()
        ret = []
        for action in actions:
            state = self.next(action)
            if not state.is_win():
                ret.append(action)
        return ret
    # 王手生成
    def check_legal_actions(self):
        actions = self.legal_actions()
        ret = []
        for action in actions:
            state = self.next(action)
            # 合法手だけに絞る
            if state.is_win():
                continue
            # 念の為historyをリセット
            # 先手後手を入れ替えて、2手指しする
            state2 = State(state.enemy_pieces.copy(),state.pieces.copy(),[])
            # 2手指しして勝てたら王手と判断
            if state2.is_win():
                ret.append(action)
        return ret
    # 駒の移動時の合法手のリストの取得
    def legal_actions_pos(self, position_src):
        actions = []

        # 駒の移動可能な方向
        piece_type = self.pieces[position_src]
        if piece_type > 4: piece_type-4
        directions = []
        if piece_type == 1: # ヒヨコ
            directions = [0]
        elif piece_type == 2: # ゾウ
            directions = [1, 3, 5, 7]
        elif piece_type == 3: # キリン
            directions = [0, 2, 4, 6]
        elif piece_type == 4: # ライオン
            directions = [0, 1, 2, 3, 4, 5, 6, 7]

        # 合法手の取得
        for direction in directions:
            # 駒の移動元
            x = position_src%3 + self.dxy[direction][0]
            y = int(position_src/3) + self.dxy[direction][1]
            p = x + y * 3

            # 移動可能時は合法手として追加
            if 0 <= x and x <= 2 and 0<= y and y <= 3 and self.pieces[p] == 0:
                actions.append(self.position_to_action(p, direction))
        return actions

    # 次の状態の取得
    def next(self, action):
        next_hist = self.history.copy()
        next_hist.append(self.hash_key())
        # 次の状態の作成
        state = State(self.pieces.copy(), self.enemy_pieces.copy(), next_hist)

        # 行動を(移動先, 移動元)に変換
        position_dst, position_src = self.action_to_position(action)

        # 駒の移動
        if position_src < 8:
            # 駒の移動元
            x = position_dst%3 - self.dxy[position_src][0]
            y = int(position_dst/3) - self.dxy[position_src][1]
            position_src = x + y * 3

            # 駒の移動
            state.pieces[position_dst] = state.pieces[position_src]
            state.pieces[position_src] = 0

            # 相手の駒が存在する時は取る
            piece_type = state.enemy_pieces[11-position_dst]
            if piece_type != 0:
                if piece_type != 4:
                    state.pieces[11+piece_type] += 1 # 持ち駒+1
                state.enemy_pieces[11-position_dst] = 0

        # 持ち駒の配置
        else:
            capture = position_src-7
            state.pieces[position_dst] = capture
            state.pieces[11+capture] -= 1 # 持ち駒-1

        # 駒の交代
        w = state.pieces
        state.pieces = state.enemy_pieces
        state.enemy_pieces = w
        return state

    # 先手かどうか
    def is_first_player(self):
        return len(self.history) % 2 == 0

    def hash_key(self):
        key = turn_key[0] if self.is_first_player() else turn_key[1]
        pieces0 = self.pieces  if self.is_first_player() else self.enemy_pieces
        pieces1 = self.enemy_pieces  if self.is_first_player() else self.pieces
        # 持ち駒
        for i in range(12, 15):
            key ^= enemy_hand_piece_key[i-11][pieces1[i]]
            key ^= self_hand_piece_key[i-11][pieces0[i]]
        # ボード
        for i in range(12):
            key ^= self_piece_key[pieces0[i]][i]
            key ^= enemy_piece_key[pieces1[11-i]][11-i]
        return key
    # actionの文字列化
    def action_str(self, action):
        if action == -1:
            return "NONE"
        position_dst, position_src = self.action_to_position(action)
        if position_src < 8:
            # 駒の移動元
            x = position_dst%3 - self.dxy[position_src][0]
            y = int(position_dst/3) - self.dxy[position_src][1]
            position_src = x + y * 3
            return f"指({position_src}, {position_dst})"
        else:
            hzkr = ('', 'ひ', 'ぞ', 'き', 'ら')
            capture = position_src-7
            return f"打({hzkr[capture]}, {position_dst})"

    # 文字列表示
    def __str__(self):
        pieces0 = self.pieces  if self.is_first_player() else self.enemy_pieces
        pieces1 = self.enemy_pieces  if self.is_first_player() else self.pieces
        hzkr1 = ('', 'vひ', 'vぞ', 'vき', 'vら')
        hzkr0 = ('', '^ひ', '^ぞ', '^き', '^ら')
        s = "先\n" if self.is_first_player() else "後\n"
        s += f"depth:{len(self.history)}\n"
        s += f"hash:{self.hash_key()}\n"
        for i in range(2):
            s += "["
            for j in range(15):
                p = self.pieces if i == 0 else self.enemy_pieces
                s += str(p[j])
                if j != 14:
                    s += ","
            s += "]\n"

        # 後手の持ち駒
        s += '['
        for i in range(12, 15):
            if pieces1[i] >= 2: s += hzkr1[i-11]
            if pieces1[i] >= 1: s += hzkr1[i-11]
        s += ']\n'

        # ボード
        for i in range(12):
            if pieces0[i] != 0:
                s += hzkr0[pieces0[i]]
            elif pieces1[11-i] != 0:
                s += hzkr1[pieces1[11-i]]
            else:
                s += ' - '
            if i % 3 == 2:
                s += '\n'

        # 先手の持ち駒
        s += '['
        for i in range(12, 15):
            if pieces0[i] >= 2: s += hzkr0[i-11]
            if pieces0[i] >= 1: s += hzkr0[i-11]
        s += ']\n'
        return s

# ランダムで行動選択
def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions)-1)]

def eval(state):
    material = (0, 100, 400, 400, 5000)
    score = 0
    pieces0 = state.pieces 
    pieces1 = state.enemy_pieces 
    # 持ち駒
    for i in range(12, 15):
        score += pieces0[i] * material[i-11]
        score -= pieces1[i] * material[i-11]
    # ボード
    for i in range(12):
        score += material[pieces0[i]]
        score -= material[pieces1[i]]
    return score + random.randint(-10,10)

# アルファベータ法で状態価値計算
def alpha_beta(state, alpha, beta, depth):
    # 負けは状態価値-1
    if state.is_lose():
        return -8000

    # 引き分けは状態価値0
    if state.is_draw():
        return  0
    
    if state.is_win():
        return 8000
    
    if depth <= 0:
        return eval(state)

    # 合法手の状態価値の計算
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -beta, -alpha, depth-1)
        if score > alpha:
            alpha = score

        # 現ノードのベストスコアが親ノードを超えたら探索終了
        if alpha >= beta:
            return alpha

    # 合法手の状態価値の最大値を返す
    return alpha

# アルファベータ法で行動選択
def alpha_beta_action5(state):
    # 合法手の状態価値の計算
    best_action = 0
    alpha = -float('inf')
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -float('inf'), -alpha, 5)
        if score > alpha:
            best_action = action
            alpha = score

    # 合法手の状態価値の最大値を持つ行動を返す
    #print(f"best_score:{alpha}")
    return best_action

# アルファベータ法で行動選択
def alpha_beta_action4(state):
    # 合法手の状態価値の計算
    best_action = 0
    alpha = -float('inf')
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -float('inf'), -alpha, 4)
        if score > alpha:
            best_action = action
            alpha = score

    # 合法手の状態価値の最大値を持つ行動を返す
    #print(f"best_score:{alpha}")
    return best_action

# アルファベータ法で行動選択
def alpha_beta_action3(state):
    # 合法手の状態価値の計算
    best_action = 0
    alpha = -float('inf')
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -float('inf'), -alpha, 2)
        if score > alpha:
            best_action = action
            alpha = score

    # 合法手の状態価値の最大値を持つ行動を返す
    #print(f"best_score:{alpha}")
    return best_action

# アルファベータ法で行動選択
def alpha_beta_action2(state):
    # 合法手の状態価値の計算
    best_action = 0
    alpha = -float('inf')
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -float('inf'), -alpha, 2)
        if score > alpha:
            best_action = action
            alpha = score

    # 合法手の状態価値の最大値を持つ行動を返す
    #print(f"best_score:{alpha}")
    return best_action

# アルファベータ法で行動選択
def alpha_beta_action1(state):
    # 合法手の状態価値の計算
    best_action = 0
    alpha = -float('inf')
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -float('inf'), -alpha, 1)
        if score > alpha:
            best_action = action
            alpha = score

    # 合法手の状態価値の最大値を持つ行動を返す
    #print(f"best_score:{alpha}")
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

def test():
    state = State([0,0,0,0,0,0,0,2,1,0,4,3,0,0,0],[0,0,0,0,0,0,0,4,1,2,0,3,0,0,0],[])
    print(state)
    l = state.perfect_legal_actions()
    print("---------")
    for ll in l:
        print(state.action_str(ll))
# 動作確認
if __name__ == '__main__':
    is_end_num = 0
    is_draw_num = 0
    sum_depth = 0
    init_key()
    #test()
    #exit(1)
    
    while True:
        # 状態の生成
        state = State()
        ar = state.pieces_array()
        print(f"draw:{is_draw_num} end:{is_end_num} \r",end="")
        # ゲーム終了までのループ
        while True:
            #print(stateW)
            # ゲーム終了時
            if state.is_lose():
                is_end_num += 1
                break
            if state.is_draw():
                is_draw_num += 1
                break
            # 次の状態の取得
            legal_actions = state.perfect_legal_actions()
            
            if len(legal_actions) == 0:
                is_end_num += 1
                break
            next_action = legal_actions[random.randint(0, len(legal_actions)-1)]
            state = state.next(next_action)

            # 文字列表示
            #print(state)
            #print()
        