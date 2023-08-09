from single_network import *
import json
import random
def random_state():
    state = State()
    max_ply = random.randint(0,9)
    for i in range(max_ply):
        legal_actions = state.legal_actions()
        m = legal_actions[random.randint(0, len(legal_actions)-1)]  
        state = state.next(m)
    return state

def preudo_data():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SingleNet()
    model.load_state_dict(torch.load('./model/best_single.h5'))
    model = model.to(device)
    model.eval()
    output_list = []
    for i in range(1000):
        state = random_state()
        feature = np.array(state.feature())
        file, rank, channel = DN_INPUT_SHAPE
        feature = feature.reshape(1, channel, file, rank)
        f = torch.from_numpy(feature)
        f = f.to(device).float()
        output = model(f)
        output_list.append({ "p": state.hash_key(), "s": output.item(), "r": 0})
    json_str = json.dumps(output_list)
    with open(f"data/pseudo.json", mode='w') as f:
        f.write(json_str)
if __name__ == '__main__':
    preudo_data()