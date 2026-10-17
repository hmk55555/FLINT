"""Helpers shared by the scripts."""

import glob
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def find_latest_checkpoint(experiment_root):
    import torch

    best_ckpt, best_step = None, -1
    for path in glob.glob(os.path.join(experiment_root, "**/checkpoints/colbert*/optimizer_state.pt"), recursive=True):
        step = torch.load(path, map_location="cpu")["batch"]
        if step > best_step:
            best_ckpt, best_step = os.path.dirname(path), step
    return best_ckpt


def require_files(*paths):
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing training data file: {p} (see README, 'Data').")


def count_lines(path):
    with open(path) as f:
        return sum(1 for _ in f)
