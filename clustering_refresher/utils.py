import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List


def generate_data(
    seed: int,
) -> pd.DataFrame:
    np.random.seed(seed)
    dist1 = np.random.normal(loc=[5, 5], scale=2, size=(100, 2))
    dist2 = np.random.normal(loc=[2, 2], scale=1, size=(100, 2))
    dist3 = np.random.normal(loc=[3, 4], scale=2.5, size=(100, 2))

    df = pd.DataFrame(
        np.vstack([dist1, dist2, dist3]),
        columns=["X", "Y"],
    )

    return df


COLORS = [
    "mediumslateblue",
    "darkorange",
    "chartreuse",
    "crimson",
    "khaki",
    "slategrey",
    "deeppink",
    "bisque",
    "orchid",
    "mediumvioletred",
    "mediumpurple",
    "salmon",
    "turquoise",
]


def generate_clusters_plot(
    df: pd.DataFrame,
    title: str,
    output_filename: str,
    centroids: np.ndarray = None,
    include_noise: bool = False,
    ax: Axes = None,
):
    n_clusters = df["labels"].nunique()
    if n_clusters > len(COLORS):
        raise IndexError(
            f"Insufficient number of colors ({len(COLORS)}) for the number of clusters ({n_clusters})"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    for cluster_id in range(n_clusters):
        cluster_points = df[df["labels"] == cluster_id]
        ax.scatter(
            cluster_points["X"],
            cluster_points["Y"],
            c=COLORS[cluster_id],
            edgecolors="black",
            label=f"cluster {cluster_id}",
        )

    if include_noise:
        noise_points = df[df["labels"] == -1]
        ax.scatter(
            noise_points["X"],
            noise_points["Y"],
            c="lightgray",
            label="noise",
        )

    if centroids is not None:
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="*",
            s=300,
            c="black",
            label="centroid",
        )

    ax.set_xlim([-2, 10])
    ax.set_ylim([-2, 10])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.set_aspect("equal")

    if ax is None:
        ax.legend()
        fig.savefig("output/" + output_filename)
