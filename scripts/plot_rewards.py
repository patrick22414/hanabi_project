import re
import sys

import numpy as np

import matplotlib

font = {
    "family": "Helvetica",
    "size": 20,
}

matplotlib.rc("font", **font)

from matplotlib import pyplot as plt

assert len(sys.argv) > 1

pattern_1 = re.compile(r"Iteration (\d*)")
pattern_2 = re.compile(r"avg_reward=(\d*\.\d*),")

labels = None
labels = [
    "MLP128 + MINIMAL obs.",
    "RNN128 + MINIMAL obs.",
    # "MLP128 #3",
    # "MLP128 #4",
    # "MLP128 #5",
]

plt.figure(figsize=(8, 6))

for i, filename in enumerate(sys.argv[1:]):
    iteration = 0
    iterations = []
    rewards = []

    with open(filename, "r") as logfile:
        for line in logfile:
            match = pattern_1.search(line)
            if match:
                iteration = int(match.group(1))

            match = pattern_2.search(line)
            if match:
                iterations.append(iteration)
                reward = float(match.group(1))
                rewards.append(reward)

    rewards = np.convolve(np.pad(rewards, (5,), "edge"), np.ones(11) / 11, mode="valid")

    if labels is not None:
        plt.plot(iterations, rewards, linewidth=2, label=labels[i])
    else:
        plt.plot(iterations, rewards, linewidth=2, label=filename)

plt.xlabel("Iterations")
plt.xlim(0, 2000)

plt.ylabel("Score (11-point average)")
plt.ylim(0, 10)

plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.savefig("rewards.png")
plt.savefig("rewards.pdf")
