import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import generate_data, generate_clusters_plot

# from sklearn import metrics

# Set model params.
N_CLUSTERS = 4
INIT = "k-means++"
MAX_ITER = 300
RNG_SEED = 2025

# Generate sample data.
df = generate_data(RNG_SEED)

# Fit the model.
model = KMeans(
    n_clusters=N_CLUSTERS, init=INIT, max_iter=MAX_ITER, random_state=RNG_SEED
)
model.fit(df)

# Fetch the cluster centres.
centroids = model.cluster_centers_
centroids_df = pd.DataFrame(centroids, columns=["X", "Y"])

# Fetch the assigned cluster for each point.
labels = model.labels_
df["labels"] = labels

# Plot the clusters and the centroids.
generate_clusters_plot(
    df,
    "Clusters",
    output_filename="KMean_clusters.png",
    centroids=centroids,
)


# Try figuring out the optimal number of clusters with the Elbow Method.
# Should be 3 ideally.
def elbow_method(df):
    inertia = []
    for n_clusters in range(1, 10):
        model = KMeans(n_clusters=n_clusters, random_state=RNG_SEED)
        model.fit(df)
        inertia.append(model.inertia_)

    plt.figure(figsize=(17, 8))
    plt.plot(range(1, 10), inertia, "bx-")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.title("Kmeans: elbow method")
    plt.savefig("output/kmeans_elbow.png")


elbow_method(df)
