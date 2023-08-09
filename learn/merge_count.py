import json
from pathlib import Path
def merge_count():
    json_list = list(Path('./').glob('count*.json'))
    count_all_dict = {}
    for path in json_list:
        with path.open(mode='r') as f:
            try:
                data_dict = json.loads(f.read())
            except json.decoder.JSONDecodeError:
                print(f"decode error:{path}")
                pass
            for key, num in data_dict.items():
                num_all = int(count_all_dict.get(key) or 0)
                num_all += num
                count_all_dict[key] = num_all
    json_str = json.dumps(count_all_dict)
    for i in range(len(json_list)):
        with open(f"count{i}.json", mode='w') as f:
            f.write(json_str)
if __name__ == "__main__":
    merge_count()