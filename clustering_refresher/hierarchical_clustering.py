import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from utils import generate_data, generate_clusters_plot

# Generate sample data.
RNG_SEED = 2025
df = generate_data(RNG_SEED)

# Generate a dendogram.
model = hierarchy.linkage(df, method="ward")
dendogram = hierarchy.dendrogram(model)
plt.title("Dendogram")
plt.xlabel("Observations")
plt.ylabel("Euclidean distances")
plt.savefig("output/hierarchical_dendogram.png")

# Use the sklearn model to generate a predefined number of clusters.
model = AgglomerativeClustering(
    n_clusters=7,
    metric="euclidean",
    linkage="ward",
)
labels = model.fit_predict(df)
df["labels"] = labels

# Plot the clusters.
generate_clusters_plot(
    df,
    "Hierarchical Clustering",
    "hierarchical_clustering.png",
)
