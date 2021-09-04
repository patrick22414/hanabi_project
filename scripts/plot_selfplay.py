import re
import sys

import matplotlib
import numpy as np

font = {
    "family": "Helvetica",
    "size": 20,
}

matplotlib.rc("font", **font)

from matplotlib import pyplot as plt

assert len(sys.argv) > 1

pattern_1 = re.compile(r"=== Self-play evaluation of (\S*) ===")
pattern_2 = re.compile(r"avg_reward=(\d*\.\d*), std_reward=(\d*\.\d*),")

fig = plt.figure(figsize=(8, 4))
ax = plt.gca()

labels = []
avgs = []
stds = []

with open(sys.argv[1], "r") as logfile:
    for line in logfile:
        match = pattern_1.search(line)
        if match:
            print(match.groups())
            checkpoint_name = match.group(1)
            labels.append(checkpoint_name)

        match = pattern_2.search(line)
        if match:
            print(match.groups())
            avgs.append(float(match.group(1)))
            stds.append(float(match.group(2)))


labels = [
    # "MLP64",
    "MLP128",
    # "MLP256",
    # "RNN64",
    "RNN128",
    # "RNN256",
]

bars = ax.barh(
    np.arange(len(labels)),
    avgs,
    xerr=stds,
    height=0.5,
    capsize=8,
    edgecolor="grey",
    color="lightgrey",
    align="center",
)
ax.bar_label(
    bars,
    labels=[f"{avg} ({std})" for avg, std in zip(avgs, stds)],
    label_type="center",
    padding=8,
)

ax.set_ylim(-0.75, 1.75)
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels)
ax.invert_yaxis()

ax.set_xlim(0, 10)
ax.set_xlabel("Scores (stdev)")

plt.tight_layout()
plt.savefig("selfplay.png")
plt.savefig("selfplay.pdf")
