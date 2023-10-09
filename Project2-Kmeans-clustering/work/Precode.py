import numpy as np
import matplotlib.pyplot as plt
import random

# Load the dataset
data = np.load('AllSamples.npy')
id = "1284"

def initial_point_idx(id, k,N):
	return np.random.RandomState(seed=(id+k)).permutation(N)[:k]

def init_point(data, idx):
    return data[idx,:]

def initial_S1(id, k):
    print("Strategy 1: k and initial points")
    i = int(id)%150 
    random.seed(i+500)
    init_idx = initial_point_idx(i,k,data.shape[0])
    init_s= init_point(data, init_idx)
    return init_s

def compute_loss(data, centroids, labels):
    """Compute the loss using the given objective function."""
    loss = 0
    for i in range(centroids.shape[0]):
        cluster_data = data[labels == i]
        loss += np.sum(np.linalg.norm(cluster_data - centroids[i], axis=1)**2)
    return loss

def kmeans(data, k, initial_centroids):
    """K-means clustering algorithm."""
    centroids = initial_centroids.copy()  # Ensure we don't modify the input
    previous_centroids = np.zeros_like(centroids)
    labels = np.zeros(data.shape[0], dtype=np.int32)
    
    while not np.allclose(centroids, previous_centroids):
        # Assignment step
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
        
        # Update step
        previous_centroids = centroids.copy()
        for i in range(k):
            centroids[i] = data[labels == i].mean(axis=0)
    
    loss = compute_loss(data, centroids, labels)
    
    return centroids, labels, loss

# Loop for k from 2 to 10
results = {}
losses = []
max_k=10
ks = list(range(2, max_k + 1))

for k in range(2, 11):
    init_s= initial_S1(id, k)
    centroids, labels, loss = kmeans(data, k, init_s)
    losses.append(loss)
    results[k] = {"centroids": centroids, "labels": labels, "loss": loss}
    print(f"For k={k}, Loss={loss}, Centroids={centroids}")

    # Visualization- Uncomment to see plots
    # plt.figure(figsize=(10, 7))
    # for i in range(k):
    #     cluster_data = data[labels == i]
    #     plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {i + 1}")
    # plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label="Centroids")
    # plt.title(f"K-means Clustering for k={k}")
    # plt.xlabel("X-coordinate")
    # plt.ylabel("Y-coordinate")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

plt.plot(ks, losses, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Loss')
plt.title('Strategy 1: Loss vs. Number of Clusters')
plt.grid(True)
plt.show()