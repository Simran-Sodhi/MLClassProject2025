'''
K-means clustering
Author: Varian Zhou
Date: 2025-04-03

'''

import numpy as np

def normalize_vectors(X):
    """
    Normalize the rows of matrix X to have unit length.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1
    return X / norms


def spherical_kmeans(X, k, max_iter=100, tol=1e-4):
    """
    Perform spherical k-means clustering on data X using cosine similarity

    Parameters:
      X: numpy.ndarray of shape (n_samples, n_features)
         The input data
      k: int
         The number of clusters
      max_iter: int, optional (default=100)
         Maximum number of iterations
      tol: float, optional (default=1e-4)
         Tolerance for convergence (not used directly here, but can be
         incorporated to check changes in centroids if needed)

    Returns:
      labels: numpy.ndarray of shape (n_samples,)
         Cluster assignment for each sample
      centroids: numpy.ndarray of shape (k, n_features)
         The final cluster centroids (normalized)
    """
    # Normalize the input data
    X_norm = normalize_vectors(X)
    n_samples, n_features = X_norm.shape
    
    # Initialize centroids by randomly selecting k samples from the data
    initial_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X_norm[initial_indices]

    # Ensure centroids are normalized
    centroids = normalize_vectors(centroids)

    # Initialize labels for each point
    labels = np.zeros(n_samples, dtype=int)

    # EM to optimize
    for iteration in range(max_iter):
        # Compute cosine similarity: since both X_norm and centroids are unit vectors,
        # the dot product equals the cosine similarity
        similarity = np.dot(X_norm, centroids.T)  # Shape: (n_samples, k)

        # Assign each point to the cluster with the highest cosine similarity
        new_labels = np.argmax(similarity, axis=1)

        # If no label changes, the algorithm converged
        if np.array_equal(new_labels, labels):
            print(f"Converged after {iteration} iterations.")
            break
        labels = new_labels

        # Update centroids
        for j in range(k):
            # Select all points that belong to cluster j
            cluster_points = X_norm[labels == j]
            if len(cluster_points) == 0:
                # If a cluster has no points, randomly reinitialize its centroid
                centroids[j] = X_norm[np.random.choice(n_samples)]
            else:
                # Compute the mean of points in the cluster
                centroid = np.mean(cluster_points, axis=0)
                # Normalize the centroid to unit length.
                norm = np.linalg.norm(centroid)
                if norm == 0:
                    centroids[j] = centroid
                else:
                    centroids[j] = centroid / norm

    return labels, centroids


# Test case
if __name__ == '__main__':
    # Create some synthetic data for demonstration
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # Generate synthetic data with 3 centers
    X, y_true = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

    # Run spherical k-means clustering with cosine similarity
    k = 3
    labels, centroids = spherical_kmeans(X, k, max_iter=100)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
    plt.title("Spherical k-Means Clustering with Cosine Similarity")
    plt.legend()
    plt.show()
