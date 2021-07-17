import logging
import os
from typing import List

import numpy as np
import torch

from ppo.log import log_main


def action_histogram(actions: List[str]):
    values, counts = np.unique(actions, return_counts=True)

    line_1, line_2 = " " * 45, " " * 45

    for v, c in zip(map(str, values), map(str, counts)):
        # size = max(len(v), len(c)) + 2
        size = 12 + 2
        line_1 += v.ljust(size, " ")
        line_2 += c.ljust(size, " ")

    return line_1 + "\n" + line_2


def linear_decay(start, end, t):
    return np.linspace(start, end, t)


def cosine_decay(start, end, t):
    return (np.cos(np.linspace(0, np.pi, t)) + 1) / 2 * (start - end) + end


def checkpoint(
    i_iter,
    iterations,
    checkpoints,
    state_dict,
    policy_lr,
    value_fn_lr,
    ppo_clip,
    entropy_coef,
):
    n = i_iter // (iterations // checkpoints)
    log_main.info(f"------ Checkpoint {n}/{checkpoints} ------")

    last_handler = log_main.handlers[-1]
    if isinstance(last_handler, logging.FileHandler):
        filename = os.path.basename(last_handler.baseFilename)
        log_main.info(f"Using logfile logs/{filename}")

        n_digits = int(np.ceil(np.log10(checkpoints + 1)))
        filename = f"checkpoints/{os.path.splitext(filename)[0]}-{n:0{n_digits}d}.pt"
        torch.save(state_dict, filename)
        log_main.info(f"State dict saved to {filename}")
    else:
        log_main.info("Not using logfile")

    log_main.info(f"Current lr for policy: {policy_lr:.2e}")
    log_main.info(f"Current lr for value_fn: {value_fn_lr:.2e}")
    log_main.info(f"Current ppo_clip: {ppo_clip:.2e}")
    log_main.info(f"Current entropy_coef: {entropy_coef:.2e}")


if __name__ == "__main__":
    v = 1
    v = cosine_decay(1, 0, 10)
    for _ in range(10):
        print(next(v))
