"""
DBSCAN = Density-Based Clustering
Dense regions = clusters, sparse regions = noise.

Params:
- min_neighbors (decides core vs border)
- epsilon = distance threshold (decides core, border vs noise)

Three types of points:
- Core: have >= min_neighbors within epsilon.
- Border: have <= min_neighbors within epsilon, but reachable from a core
point;
- Noise: not reachable from a core point.

Algorithm:
Pick unvisited point -> is_core ? expand : (is_border ? assign to nearby
cluster : point is noise). Repeat until all points have been visited.

PROS: clusters don't have to be spherical, detects noise, don't have to
choose n_clusters
CONS: choosing good epsilon, varying density clusters, not good with high-dim
data
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from utils import generate_data, generate_clusters_plot
from itertools import product

RNG_SEED = 2025

# Generate sample data.
X, _ = make_moons(n_samples=500, noise=0.05, random_state=RNG_SEED)

# Run DBSCAN.
model = DBSCAN(eps=0.3, min_samples=5, metric="euclidean")
labels = model.fit_predict(X)

# Plot the clusters.
plt.scatter(
    X[labels == 0, 0],
    X[labels == 0, 1],
    c="slategrey",
    edgecolors="black",
    s=40,
    label="Cluster 1",
)

plt.scatter(
    X[labels == 1, 0],
    X[labels == 1, 1],
    c="darkorange",
    edgecolors="black",
    s=40,
    label="Cluster 2",
)

plt.legend()
plt.savefig("output/dbscan_moon.png")

# Doing it again with the data used in the other examples.
EPSILONS = [0.9, 1, 1.1]
MIN_SAMPLES = [7, 8, 9]

df = generate_data(RNG_SEED)
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))
axes = axes.flatten()

for idx, (epsilon, min_samples) in enumerate(product(EPSILONS, MIN_SAMPLES)):
    model = DBSCAN(eps=epsilon, min_samples=min_samples, metric="euclidean")
    labels = model.fit_predict(df)
    df["labels"] = labels

    generate_clusters_plot(
        df,
        f"dbscan_eps={epsilon}_samples={min_samples}",
        "",
        include_noise=True,
        ax=axes[idx],
    )

plt.tight_layout()
plt.savefig("output/dbscan_multiple_params_6.png")
