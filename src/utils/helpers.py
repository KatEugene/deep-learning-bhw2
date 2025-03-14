import torch
import numpy as np
import random
import os
from types import SimpleNamespace


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class ConfigDict(dict):
    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            return ConfigDict(value)
        elif isinstance(value, list):
            return [ConfigDict(v) if isinstance(v, dict) else v for v in value]
        return value
