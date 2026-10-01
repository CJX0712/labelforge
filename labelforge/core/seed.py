"""全局确定性种子。"""

import random

import numpy as np


def set_all(seed: int) -> None:
    """固定所有已知随机源，保证同 seed 逐位可复现。"""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
