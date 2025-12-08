import yaml
from easydict import EasyDict as edict

def load_config(config_path):
    with open(config_path, 'r') as f:
        return edict(yaml.safe_load(f))
