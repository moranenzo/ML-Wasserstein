import numpy as np
import matplotlib.pyplot as plt
from utils import wbarycenter_clustering_1d

# Génération de données factices : 3 types de distributions
n_per_cluster = 50
support = np.linspace(0, 1, 20)
data = []

for loc in [0.2, 0.5, 0.8]:
    for _ in range(n_per_cluster):
        dist = np.exp(-((support - loc) ** 2) / (2 * 0.02))
        dist += 0.05 * np.random.rand(len(support))  # bruit
        data.append(dist)

data = np.array(data)

assignments, barycenters = wbarycenter_clustering_1d(data, support, n_clusters=3)

# Plot résultat
plt.figure(figsize=(10, 5))
for i in range(len(data)):
    plt.plot(support, data[i], color=f"C{assignments[i]}", alpha=0.3)

for i, b in enumerate(barycenters):
    plt.plot(support, b, color=f"C{i}", lw=3, label=f"Barycenter {i}")

plt.legend()
plt.title("Clustering Wasserstein 1D")
plt.grid(True)
plt.show()

plt.savefig(f"test1.png")


import numpy as np
import matplotlib.pyplot as plt
from utils import wbarycenter_clustering

# Création de données multi-dimensionnelles simulées (3 groupes)
def generate_blob(center, n=50, noise=0.01):
    x = np.linspace(0, 1, 10)
    blob = np.exp(-((x - center) ** 2) / (2 * 0.02))
    blob = np.tile(blob, (n, 1)) + noise * np.random.rand(n, len(x))
    blob /= blob.sum(axis=1, keepdims=True)
    return blob

data = np.vstack([
    generate_blob(0.2),
    generate_blob(0.5),
    generate_blob(0.8)
])

assignments, barycenters = wbarycenter_clustering(data, n_clusters=3)

# Visualisation simple : moyenne des clusters
plt.figure(figsize=(10, 5))
for i, b in enumerate(barycenters):
    plt.plot(b, label=f"Barycenter {i}")
plt.title("Barycenters multi-dim")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(f"test2.png")