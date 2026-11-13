"""Random seed utility for deterministic execution across libraries."""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> int:
    """Set random seed for python, numpy, and torch.

    Args:
        seed: Integer random seed.

    Returns:
        The integer seed value used.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
