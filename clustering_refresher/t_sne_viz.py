"""
Visualising multidimensional data with T-SNE.
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

# Load data.
digits = datasets.load_digits()
X, labels = digits.data, digits.target

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=labels,
    edgecolor="none",
    alpha=0.7,
)
plt.colorbar(scatter)
plt.title("Digits tSNE")
plt.savefig("output/digits_tsne.png")
