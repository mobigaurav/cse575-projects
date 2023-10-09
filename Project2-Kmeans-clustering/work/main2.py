import numpy as np
import matplotlib.pyplot as plt
import random

data = np.load('AllSamples.npy')
id = "1284"

def initial_point_idx2(id,k, N):
    random.seed((id+k))     
    return random.randint(0,N-1)

def average_distance_to_centroids(point, centroids):
    """Compute average distance of a point to all centroids."""
    return np.mean(np.sqrt(np.sum((point - centroids) ** 2, axis=1)))

def initialize_rest_of_centroids(first_centroid, k):
    """Get k centroids with the modified K-means++ approach."""
    centroids = [first_centroid]
    
    for _ in range(1, k):
        avg_distances = np.array([average_distance_to_centroids(point, centroids) for point in data])
        new_centroid = data[np.argmax(avg_distances)]
        centroids.append(new_centroid)
        
    return np.array(centroids)


def kmeans(data, initial_centroids):
    k = initial_centroids.shape[0]
    centroids = np.copy(initial_centroids)
    prev_labels = np.zeros(data.shape[0])
    labels = np.zeros(data.shape[0])
    
    while True:
        # Assign each data point to the closest centroid
        for i, point in enumerate(data):
            distances = np.linalg.norm(point - centroids, axis=1)
            labels[i] = np.argmin(distances)

        # Re-calculate centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            # Check if any data point is assigned to the centroid
            if np.sum(labels == i) == 0:  # Empty cluster
                # Reinitialize the centroid randomly from the data points
                new_centroids[i] = data[np.random.choice(data.shape[0])]
            else:
                new_centroids[i] = data[labels == i].mean(axis=0)

        # Break condition: Check if centroids have stopped moving
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids.copy()

    return centroids, labels


def compute_loss(data, centroids, labels):
    """Compute the K-means loss."""
    return np.sum([np.linalg.norm(data[labels == i] - centroids[i])**2 for i in range(centroids.shape[0])])

def initial_S2(id, k):
    print("Strategy 2: k and initial points")
    i = int(id) % 150
    random.seed(i + 800)
    init_idx = initial_point_idx2(i, k, data.shape[0])
    init_s= data[init_idx, :]
    return init_s

# def initial_S2(id):
#     print("Strategy 2: k and initial points")
#     i = int(id) % 150
#     random.seed(i + 800)
#     k1 = 4
#     k2 = 6
#     init_idx2 = initial_point_idx2(i, k1, data.shape[0])
#     init_s1 = data[init_idx2, :]
#     init_idx2 = initial_point_idx2(i, k2, data.shape[0])
#     init_s2 = data[init_idx2, :]
#     return k1, init_s1, k2, init_s2

# k1, init_s1, k2, init_s2 = initial_S2(1284) 
# print(f"For k={k1}, first_centroid={init_s1}, k2={k2}, first_centroid={init_s2}")

# initial_centroids_4 = initialize_rest_of_centroids(init_s1, k1)
# final_centroids_4, labels_4 = kmeans(data, initial_centroids_4)
# loss_4 = compute_loss(data, final_centroids_4, labels_4)

# initial_centroids_6 = initialize_rest_of_centroids(init_s2, k2)
# final_centroids_6, labels_6 = kmeans(data, initial_centroids_6)
# loss_6 = compute_loss(data, final_centroids_6, labels_6)

# print(f"For k=4, final centroids are:\n{final_centroids_4}\nLoss: {loss_4}\n")
# print(f"For k=6, final centroids are:\n{final_centroids_6}\nLoss: {loss_6}")

results = {}
losses = []
max_k=10
ks = list(range(2, max_k + 1))

for k in range(2, 11):
    init_s= initial_S2(id, k)
    initial_centroids = initialize_rest_of_centroids(init_s, k)
    final_centroids, labels = kmeans(data, initial_centroids)
    loss = compute_loss(data, final_centroids, labels)
    losses.append(loss)
    results[k] = {"centroids": final_centroids, "labels": labels, "loss": loss}
    print(f"For k={k}, Loss={loss}, Centroids={final_centroids}")
    # Visualization
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
plt.title('Strategy 2: Loss vs. Number of Clusters')
plt.grid(True)
plt.show()
