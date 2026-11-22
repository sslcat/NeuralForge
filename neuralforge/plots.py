"""Headless figures for the README."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BLUE, RED = "#126a87", "#ae4245"


def figures(out, before, after, test, history):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#354452", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), layout="constrained")
    x, y = test
    lim = max(2.4, abs(x).max() + 0.1)
    xx, yy = np.meshgrid(*[np.linspace(-lim, lim, 240)] * 2)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    cmap = LinearSegmentedColormap.from_list("posterior", ["#90c8d6", "#fffaf2", "#efaaa1"])
    for ax, model, title in zip(axes, (before, after), ("Before training", "After training")):
        scores, _ = model.forward(grid)
        prob = np.exp(-np.logaddexp(0, scores[:, 0] - scores[:, 1])).reshape(xx.shape)
        ax.contourf(xx, yy, prob, levels=np.linspace(0, 1, 21), cmap=cmap)
        ax.contour(xx, yy, prob, levels=[0.5], colors="#343f48", linewidths=1.2)
        for k, color, marker in [(0, BLUE, "o"), (1, RED, "^")]:
            ax.scatter(*x[y == k].T, s=15, c=color, marker=marker, label=f"Arm {k}", lw=0.4, edgecolors="white")
        ax.set(aspect="equal", title=title, xlabel="Standardized x", ylabel="Standardized y")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.savefig(out / "boundaries.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for ax, metric, title in zip(axes, ("loss", "accuracy"), ("Mean cross-entropy", "Classification accuracy")):
        for subset, color in [("train", BLUE), ("validation", RED)]:
            ax.plot([row[subset][metric] for row in history], label=subset.title(), color=color)
        ax.set(title=title, xlabel="Epoch")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)
    fig.savefig(out / "learning.png", dpi=150)
    plt.close(fig)
