import json
from pathlib import Path

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
                key = data["p"]
                num += 1
                if key in pos_dict:
                    pos_dict[key] += 1
                else:
                    pos_dict[key] = 1
    print(len(pos_dict))
    print(num)

if __name__ == '__main__':
    main()
