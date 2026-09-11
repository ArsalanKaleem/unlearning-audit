"""Global seed control.

Every script calls set_all_seeds() exactly once, at the top, with the seed
taken from the config. Nothing else in the codebase is allowed to call
random.seed / np.random.seed directly -- if you need an independent stream,
use rng_for(name) so that the stream is reproducible AND independent.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def set_all_seeds(seed: int, deterministic_torch: bool = True) -> None:
    """Seed python, numpy and (if installed) torch.

    deterministic_torch=True makes cuDNN deterministic. It costs some speed
    and it is worth it: you will re-run these experiments many times.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rng_for(name: str, seed: int) -> np.random.Generator:
    """A named, independent, reproducible numpy Generator.

    Use this for anything that must not consume from the global stream --
    e.g. bootstrap resampling should not shift the probe initialisation.

    >>> rng_for("bootstrap", 0).integers(0, 10) == rng_for("bootstrap", 0).integers(0, 10)
    True
    """
    digest = hashlib.sha256(f"{name}:{seed}".encode()).digest()
    derived = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(derived)
