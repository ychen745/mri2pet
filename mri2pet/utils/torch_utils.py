import random
import torch
import numpy as np

def init_seeds(seed=0, deterministic=False):
    """
    Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html.

    Args:
        seed (int, optional): Random seed.
        deterministic (bool, optional): Whether to set deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe


