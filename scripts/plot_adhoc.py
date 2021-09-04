import matplotlib

font = {
    "family": "Helvetica",
    "size": 16,
}

matplotlib.rc("font", **font)

import numpy as np
from matplotlib import pyplot as plt


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    # ax.grid(which="minor", color="w", linewidth=5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("white", "white"),
    threshold=None,
    **textkw
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


agents = [
    "MLP64",
    "MLP128",
    "MLP256",
    "RNN64",
    "RNN128",
    "RNN256",
]

scores = np.array(
    [
        [7.20, 3.66, 3.70, 3.64, 3.50, 3.60],
        [3.56, 7.19, 5.83, 5.85, 5.56, 5.54],
        [3.69, 5.65, 7.21, 6.52, 6.53, 6.66],
        [3.57, 5.83, 6.50, 7.19, 6.67, 6.50],
        [3.67, 5.53, 6.65, 6.61, 7.15, 6.81],
        [3.64, 5.46, 6.60, 6.67, 6.78, 7.10],
    ]
)

stdevs = np.array(
    [
        [1.53, 1.56, 1.85, 1.81, 1.80, 1.85],
        [1.55, 1.54, 1.68, 2.07, 1.88, 1.91],
        [1.91, 1.69, 1.48, 1.85, 2.07, 1.96],
        [1.84, 2.10, 1.81, 1.44, 1.65, 1.80],
        [1.80, 1.91, 1.94, 1.70, 1.43, 1.68],
        [1.85, 1.82, 2.01, 1.69, 1.77, 1.37],
    ]
)


# agents = [
#     "MLP128 #1",
#     "MLP128 #2",
#     "MLP128 #3",
#     "MLP128 #4",
#     "MLP128 #5",
# ]

# scores = np.array(
#     [
#         [6.90, 5.98, 4.06, 5.68, 5.31],
#         [5.89, 7.43, 4.19, 6.22, 5.55],
#         [3.96, 4.25, 6.52, 4.01, 5.45],
#         [5.70, 6.32, 4.01, 7.31, 5.08],
#         [5.27, 5.52, 5.33, 5.19, 7.36],
#     ]
# )

# stdevs = np.array(
#     [
#         [1.30, 2.20, 1.34, 2.04, 1.95],
#         [2.30, 1.53, 1.45, 2.13, 2.01],
#         [1.41, 1.44, 1.32, 1.40, 1.86],
#         [1.96, 2.22, 1.38, 1.49, 2.21],
#         [1.86, 2.10, 1.83, 2.28, 1.48],
#     ]
# )

fig = plt.figure(figsize=(12, 8))
ax = plt.gca()

im, cbar = heatmap(scores, agents, agents, ax=ax, vmin=0, vmax=10)
text = annotate_heatmap(im, scores)

plt.tight_layout()
plt.savefig("adhoc_scores.png")
plt.savefig("adhoc_scores.pdf")

fig = plt.figure(figsize=(12, 8))
ax = plt.gca()

im, cbar = heatmap(stdevs, agents, agents, ax=ax, vmin=0, vmax=10)
text = annotate_heatmap(im, stdevs)

plt.tight_layout()
plt.savefig("adhoc_stdevs.png")
plt.savefig("adhoc_stdevs.pdf")
