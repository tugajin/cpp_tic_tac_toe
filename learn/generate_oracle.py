import json
from game import *

def mate_search(state):
    if state.is_lose():
        return -1
    elif state.is_draw():
        return 0
    
    legal_moves = state.legal_actions()
    best_score = -2
    for i in range(len(legal_moves)):
        action = legal_moves[i]
        new_state = state.next(action)
        score = -mate_search(new_state)
        if score == 1:
            return score
        if score > best_score:
            best_score = score
    return best_score

def main():
    with open('all_pos.json') as f:
        all_pos_list = json.load(f)
    oracle_list = []
    for pos_key in all_pos_list:
        state = from_hash(pos_key)
        score = mate_search(state)
        oracle_list.append({"p":pos_key,"r":score})
    with open('oracle_result.json',"w") as f:
        json.dump(oracle_list,f)

if __name__ == '__main__':
    main()