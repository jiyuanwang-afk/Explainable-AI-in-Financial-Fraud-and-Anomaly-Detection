import os, json, random, numpy as np, torch

def set_seed(seed: int = 7):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
