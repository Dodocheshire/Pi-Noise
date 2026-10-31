"""Keep evaluation draws repeatable without perturbing the training RNG stream."""

import random
from contextlib import contextmanager

import numpy as np
import torch


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextmanager
def seeded(seed: int):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    with torch.random.fork_rng():
        seed_all(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)
