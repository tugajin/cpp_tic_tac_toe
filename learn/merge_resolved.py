import json
from pathlib import Path
import os
def merge_resolved():
    json_list = list(Path('./data').glob('resolved*.json'))
    resolved_all_dict = {}
    for path in json_list:
        with path.open(mode='r') as f:
            try:
                data_list = json.loads(f.read())
            except json.decoder.JSONDecodeError:
                print(f"decode error:{path}")
                pass
            for data in data_list:
                key = data["p"]
                r = data["r"]
                s = data["s"]
                if not key in resolved_all_dict:
                    resolved_all_dict[key] = data
        os.remove(path)
    resolved_all_list = [ obj for obj in resolved_all_dict.values() ]
    json_str = json.dumps(resolved_all_list)
    with open(f"data/const.json", mode='w') as f:
        f.write(json_str)
if __name__ == "__main__":
    merge_resolved()