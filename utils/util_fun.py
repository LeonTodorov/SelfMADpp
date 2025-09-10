import yaml

def load_data_config(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def grouped_powerset(items, return_empty = True):
    subsets = [
        [elem for i, x in enumerate(items) if (mask >> i) & 1 for elem in (x if isinstance(x, list) else [x])]
        for mask in range(1 << len(items))
    ]
    if not return_empty:
        subsets = [s for s in subsets if s]
    return subsets