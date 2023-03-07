import os
from pathlib import Path
import random
import toml

import numpy as np
import torch
import logging


PROJECT_PATH = Path(os.path.dirname(__file__)).parents[0]
EXPERIMENTS_PATH = Path(PROJECT_PATH / "experiments")

def load_experiment_config(experiment_name):
    config = toml.load(EXPERIMENTS_PATH / f"{experiment_name}.toml")
    return config

def prep_experiment_dir(experiment):
    experiment_base_dir = os.path.join(EXPERIMENTS_PATH, str(experiment))
    try:
        os.mkdir(experiment_base_dir)
    except FileExistsError as _:
        pass
    try:
        experiment_idx = max([int(max(n.lstrip("0"), "0")) for n in os.listdir(experiment_base_dir)]) + 1
    except ValueError as _:
        experiment_idx = 0
    experiment_dir = os.path.join(experiment_base_dir, str(experiment_idx).zfill(3))
    try:
        os.mkdir(experiment_dir)
    except FileExistsError as _:
        pass
    return Path(experiment_dir)

def init_seeds(seed):
    # Randomly seed pytorch to get random model weights.
    # Don't use torch.seed() because of
    # https://github.com/pytorch/pytorch/issues/33546
    torch.manual_seed(seed)
    # Make rest of experiments deterministic (almost, see
    # https://pytorch.org/docs/stable/notes/randomness.html)
    random.seed(seed)
    np.random.seed(seed)
    
def init_logger(filename):
    logging.basicConfig(filename=filename, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())