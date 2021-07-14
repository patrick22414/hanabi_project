from typing import List

import numpy as np


def action_histogram(actions: List[str]):
    values, counts = np.unique(actions, return_counts=True)

    line_1, line_2 = " " * 45, " " * 45

    for v, c in zip(map(str, values), map(str, counts)):
        # size = max(len(v), len(c)) + 2
        size = 12 + 2
        line_1 += v.ljust(size, " ")
        line_2 += c.ljust(size, " ")

    return line_1 + "\n" + line_2


def linear_decay(start, t, end=0.0):
    for value in np.linspace(start, end, t):
        yield value


if __name__ == "__main__":
    v = 1.0
    for x in linear_decay(v, t=10, end=0.1):
        print(x)
