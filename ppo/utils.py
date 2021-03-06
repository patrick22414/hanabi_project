import logging
import os
import sys
import time
from datetime import datetime
from typing import List

import numpy as np
import torch

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

FORMAT = "%(asctime)sZ %(levelname)-7s %(name)-11s %(message)s"
DATEFORMAT = "%Y-%m-%dT%H:%M:%S"

sh = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter(fmt=FORMAT, datefmt=DATEFORMAT)
formatter.converter = time.gmtime
sh.setFormatter(formatter)

loggers = [
    logging.getLogger("ppo.main"),
    logging.getLogger("ppo.collect"),
    logging.getLogger("ppo.train"),
    logging.getLogger("ppo.eval"),
]
for logger in loggers:
    logger.setLevel(logging.INFO)
    logger.addHandler(sh)


def set_logfile(prefix, filename="time"):
    if filename == "none":
        return None

    if filename == "time":
        filename = f"logs/{prefix}_{datetime.utcnow():%Y-%m-%dT%H%MZ}.log"
    else:
        filename = f"logs/{filename}"

    fh = logging.FileHandler(filename, mode="w", delay=True)
    fh.setFormatter(formatter)
    for logger in loggers:
        logger.addHandler(fh)

    return filename


def score_histogram(scores: List[str]):
    values, counts = np.unique(scores, return_counts=True)

    line_1, line_2 = " " * 4, " " * 4

    for v, c in zip(map(str, values), map(str, counts)):
        line_1 += v.rjust(4, " ")
        line_2 += c.rjust(4, " ")

    return line_1 + "\n" + line_2


def action_histogram(actions: List[str]):
    values, counts = np.unique(actions, return_counts=True)

    line_1, line_2 = " " * 4, " " * 4

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


log_main = logging.getLogger("ppo.main")


def checkpoint(
    i_iter,
    iterations,
    checkpoints,
    state_dict,
    lr,
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

    log_main.info(f"Current learning rate: {lr:.2e}")
    log_main.info(f"Current ppo_clip: {ppo_clip:.2e}")
    log_main.info(f"Current entropy_coef: {entropy_coef:.2e}")


if __name__ == "__main__":
    v = 1
    v = cosine_decay(1, 0, 10)
    for _ in range(10):
        print(next(v))
