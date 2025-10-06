import random
import torch
import numpy as np
import time

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

def time_sync():
    """Return PyTorch-accurate time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def intersect_dicts(da, db, exclude=()):
    """
    Return a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values.

    Args:
        da (dict): First dictionary.
        db (dict): Second dictionary.
        exclude (tuple, optional): Keys to exclude.

    Returns:
        (dict): Dictionary of intersecting keys with matching shapes.
    """
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}