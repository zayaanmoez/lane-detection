import yaml
import tensorflow as tf
from attrdict import AttrDict

CONFIG_PATH = 'config.yaml'

def get_config():
    with open(CONFIG_PATH, 'r') as f:
        config = AttrDict(yaml.load(f, Loader=yaml.FullLoader))
    return config