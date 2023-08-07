import gamelibs
import random

pos_dict = {}

MOVE_NONE = gamelibs.Move.MOVE_NONE

class MoveList:
    def __init__(self, ml = None):
        self.ml = gamelibs.MoveList() if ml is None else ml
    def add(self,move):
        self.ml.add(move)
    def __getitem__(self, i):
        return self.ml[i]
    def __str__(self):
        return self.ml.__str__()
    def __len__(self):
        return self.ml.len()

class State:
    def __init__(self, pos = None):
        self.pos = gamelibs.hirate() if pos is None else pos
    def turn(self):
        return self.pos.turn()
    def next(self, action):
        return State(self.pos.next(action))
    def __str__(self):
        return self.pos.__str__()
    def is_win(self):
        return self.pos.is_win()
    def is_draw(self):
        return self.pos.is_draw()
    def is_lose(self):
        return self.pos.is_lose()
    def is_done(self):
        return self.pos.is_done()
    def hash_key(self):
        return gamelibs.hash_key(self.pos)
    def legal_actions(self):
        ml = gamelibs.MoveList()
        gamelibs.legal_moves(self.pos,ml)
        return MoveList(ml)
    def feature(self):
        return gamelibs.feature(self.pos)
    def mate_search(self, ply):
        return gamelibs.mate_search(self.pos, ply)
    def in_checked(self):
        return gamelibs.in_checked(self.pos)
def from_hash(h):
    return State(gamelibs.from_hash(h))

def val_to_move(v):
    return gamelibs.val_to_move(v)

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
   
def gen_pos_list():
    history = []
    # 状態の生成
    for i in range(100000000000000000):
        state = State()
        print(f"\rtry:{i} num:{len_pos_dict()}",end="")
        #ゲーム終了までのループ
        while True:
            if state.hash_key() not in pos_dict:
                history.append([0])
            append_pos_dict(state.hash_key())
            # ゲーム終了時
            if state.is_done():
                break
            if not state.in_checked():
                action = state.mate_search(5)
            else:
                action = MOVE_NONE
            if (action != MOVE_NONE):
                m = action
            else:
                legal_actions = state.legal_actions()
                if (len(legal_actions) == 0):
                    break
                m = legal_actions[random.randint(0, len(legal_actions)-1)]
            # 次の状態の取得
            #print(state)
            state = state.next(m)
    print(f"history_len:{len(history)}")
    #write_data(history)

if __name__ == "__main__":
    #state = State()
    #print(state)
    gen_pos_list()