import json
from pathlib import Path
from game import *

def main():
    json_list = list(Path('./data').glob('*.json'))
    pos_dict = {}
    num = 0
    for path in json_list:
        with path.open(mode='r') as f:
            try:
                data_list = json.loads(f.read())
            except json.decoder.JSONDecodeError:
                print(f"decode error:{path}")
                pass
            for data in data_list:
                state = from_hash(data["p"])
                mirror = state.mirror()
                rotate90 = state.rotate()
                mirror90 = rotate90.mirror()
                rotate180 = rotate90.rotate()
                mirror180 = rotate180.mirror()
                rotate270 = rotate180.rotate()
                mirror270 = rotate180.mirror()
                
                l = [ str(state.history()),
                    str(mirror.history()),
                    str(rotate90.history()),
                    str(rotate90.history()),
                    str(rotate180.history()),
                    str(rotate180.history()),
                    str(rotate270.history()),
                    str(mirror270.history())
                    ]
                print("---------------------------------------------")
                print(":".join(l))
                print(state)
                print("s:", data["s"])
                print("r:", data["r"])

if __name__ == '__main__':
    main()
