import os
import random

import numpy as np
import torch


def set_random_seed(seed=42):
    if seed == 'None':
        return -1
    else:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = False
        return 0
