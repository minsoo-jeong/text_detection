import torch
import numpy as np
import re
import random

from collections import OrderedDict

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def filter_word(word, pattern='[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z\s]'):
    word = word.replace(" ", "").lower()
    pattern = re.compile(pattern)
    filtered = re.sub(pattern, '', word)

    return filtered



def remove_module_prefix(state_dict):

    _state_dict= OrderedDict()

    for k,v in state_dict.items():
        _state_dict[k.replace('module.','')]=v
    return _state_dict
