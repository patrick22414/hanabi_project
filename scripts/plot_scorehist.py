import matplotlib

font = {
    "family": "Helvetica",
    "size": 20,
}

matplotlib.rc("font", **font)

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(7, 6))
ax = plt.gca()

scores = """
       1   2   3   4   5   6   7   8   9  10
       7   2  24  24  54 121 270 324 145  29
"""

xs = list(map(int, "       1   2   3   4   5   6   7   8   9  10".split()))
ys = list(map(int, "       7   2  24  24  54 121 270 324 145  29".split()))

ax.bar(xs, ys)

ax.set_xticks(xs)
ax.set_xlabel("Scores")
ax.set_ylabel("Occurances")

plt.tight_layout()
plt.savefig("scorehist.pdf")
