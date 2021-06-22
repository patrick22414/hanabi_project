from typing import List

import numpy as np


def action_histogram(actions: List[str]):
    values, counts = np.unique(actions, return_counts=True)

    line_1, line_2 = "\t", "\t"

    for v, c in zip(map(str, values), map(str, counts)):
        size = max(len(v), len(c)) + 2
        line_1 += v.ljust(size, " ")
        line_2 += c.ljust(size, " ")

    return line_1 + "\n" + line_2
