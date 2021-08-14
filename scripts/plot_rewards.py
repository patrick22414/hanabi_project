import re
import sys

import numpy as np
from matplotlib import pyplot as plt

assert len(sys.argv) > 1

pattern = re.compile(r"avg_reward=(\d*\.\d*),")
pattern_iter = re.compile(r"Iteration (\d*)")

plt.figure()

for filename in sys.argv[1:]:
    iteration = 0
    iterations = []
    rewards = []

    with open(filename, "r") as logfile:
        for line in logfile:
            match = pattern_iter.search(line)
            if match:
                iteration = int(match.group(1))

            match = pattern.search(line)
            if match:
                iterations.append(iteration)
                reward = float(match.group(1))
                rewards.append(reward)

    rewards = np.convolve(np.pad(rewards, (2,), "edge"), np.ones(5) / 5, mode="valid")

    plt.plot(iterations, rewards, label=filename)

plt.ylim(0, 10)
plt.legend()
plt.grid()
plt.savefig("rewards.png")
