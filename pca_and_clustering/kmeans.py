import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from K_means import spherical_kmeans
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import os


# Define output directory
baseline_save_directory = './output'
os.makedirs(baseline_save_directory, exist_ok=True)


def classify_sample(sample):
    """
        Classify a sample into 'Normal', 'Metastatic', 'Primary', or NaN.

        Parameters
        ----------
        sample : Sample type description.

        Returns
        -------
        Classified label ('Normal', 'Metastatic', 'Primary') or np.nan if not reported.
        """
    if "Normal" in sample:
        return "Normal"
    if "Metastatic" in sample:
        return "Metastatic"
    elif sample == "Not Reported":
        return np.nan
    else:
        return "Primary"


def normalize_rows(A):
    """
    Row-normalize a matrix to unit L2 norm.

    Parameters
    ----------
    A : Input matrix of shape (n_samples, n_features).

    Returns
    -------
    Row-normalized matrix with the same shape as A.
    """
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return A / norms

def compute_avg_inertia(X, labels, centroids):
    """
    Compute average within-cluster squared cosine-distance

    Parameters
    ----------
    X : Input data matrix (n_samples × n_features)
    labels : Cluster assignments for each sample (shape: n_samples)
    centroids : Cluster centroids (shape: n_clusters × n_features)

    Returns
    -------
    Average squared cosine-distance (inertia)
    """
    Xn = normalize_rows(X)
    Cn = normalize_rows(centroids)
    dots = Xn.dot(Cn.T)  # (n_samples, k)
    assigned = dots[np.arange(X.shape[0]), labels]
    dists = 1.0 - assigned
    return np.mean(dists**2)

def elbow_plot(X, ks, max_iter=100):
    """
    Plot the elbow curve for k-means clustering.
    Runs k-means over different numbers of clusters, computes the average squared cosine-distance
    (inertia) for each, and plots the results. Saves the plot to './output/elbow_plot.png'

    Parameters
    ----------
    X : Input data matrix (n_samples × n_features), L2-normalized
    ks : List of cluster numbers to try
    max_iter : Maximum iterations for k-means (default=100)

    Returns
    -------
    inertias : Average inertia values for each k
    """
    inertias = []
    for k in ks:
        labels, centroids = spherical_kmeans(X, k, max_iter=max_iter)
        inertias.append(compute_avg_inertia(X, labels, centroids))
        print(f"k={k}, avg inertia={inertias[-1]:.4f}")
    plt.figure(figsize=(6,4))
    plt.plot(ks, inertias, 'o-', color='tab:blue')
    plt.xticks(ks)
    plt.xlabel("Number of clusters $k$")
    plt.ylabel("Average squared cosine‑distance")
    plt.title("Elbow Method for Spherical k‑Means")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('./output/elbow_plot.png')
    plt.show()
    return inertias





# --- Load & QC ---
# Only keep cells with at least 200 genes expressed
# Only keep genes expressed in at least 10 cells
adata = sc.read_h5ad('./tpm_unstranded_subset_corrected.h5ad')
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=10)
X = adata.X  # raw matrix (cells × genes)

# Standardization
X = sklearn.preprocessing.StandardScaler().fit_transform(X)




# --- Elbow analysis ---
ks = list(range(1, 11))
elbow_plot(X, ks, max_iter=100)

# Choose k from the results of elbow analysis, here we only use k=3 as illustration.
k = 3
labels, centroids = spherical_kmeans(X, k, max_iter=100)




# --- PCA visualization ---
# 2D-PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
centroids_pca = pca.transform(centroids)

# 3D-PCA
pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X)
print(pca_3d.explained_variance_ratio_)
centroids_pca_3d = pca_3d.transform(centroids)




# --- Ground truth labeling ---
true_labels = adata.obs['Sample Type']
true_labels = true_labels.map(classify_sample)

# Encode target labels for the following visualizations.
le = LabelEncoder()
true_labels_enc = le.fit_transform(true_labels)
unique_labels = le.classes_
colors = plt.cm.viridis(np.linspace(0,1,len(unique_labels)))
patches = [mpatches.Patch(color=colors[i], label=unique_labels[i])
           for i in range(len(unique_labels))]

# Get the explained variance ratio of each PC for following visualization.
explained = pca.explained_variance_ratio_
explained_3d = pca_3d.explained_variance_ratio_



# ---  Create a figure with a grid of PCA results ---
fig = plt.figure(figsize=(20, 20))
# Create a gridspec with 2 rows, 4 columns.
gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)

##########################
# --- Row 1: k-Means Results
##########################

# (0,0): 2D PCA of k-Means
ax_2dPCA = fig.add_subplot(gs[0, 0])
sc = ax_2dPCA.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
ax_2dPCA.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='x', s=100, label='Centroids')
ax_2dPCA.set_title("2D PCA of Spherical k‑Means (k=%d)" % k)
ax_2dPCA.legend()
ax_2dPCA.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
ax_2dPCA.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")


# (0,1): 3D PCA of k-Means
ax_3dPCA = fig.add_subplot(gs[0, 1], projection='3d')
ax_3dPCA.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                 c=labels, cmap='viridis', s=50, alpha=0.6)
ax_3dPCA.scatter(centroids_pca_3d[:, 0], centroids_pca_3d[:, 1], centroids_pca_3d[:, 2],
                 c='red', marker='x', s=100, label='Centroids')
ax_3dPCA.set_xlabel("PC1")
ax_3dPCA.set_ylabel("PC2")
ax_3dPCA.set_zlabel("PC3")
ax_3dPCA.set_title("3D PCA of Spherical k‑Means (k=%d)" % k)
ax_3dPCA.legend()

ax_3dPCA.set_xlabel(f"PC1 ({explained_3d[0]*100:.1f}%)")
ax_3dPCA.set_ylabel(f"PC2 ({explained_3d[1]*100:.1f}%)")
ax_3dPCA.set_zlabel(f"PC3 ({explained_3d[2]*100:.1f}%)")


##########################
# --- Row 2: Ground Truth Plots
##########################

# (1,0): 2D PCA with Ground Truth
ax_gt2dPCA = fig.add_subplot(gs[1, 0])
ax_gt2dPCA.scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels_enc, cmap='viridis', alpha=0.6)
ax_gt2dPCA.set_title("2D PCA of Ground Truth")
ax_gt2dPCA.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

ax_gt2dPCA.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
ax_gt2dPCA.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")

# (1,1): 3D PCA with Ground Truth
ax_gt3dPCA = fig.add_subplot(gs[1, 1], projection='3d')
ax_gt3dPCA.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                   c=true_labels_enc, cmap='viridis', s=50, alpha=0.6)
ax_gt3dPCA.set_xlabel("PC1")
ax_gt3dPCA.set_ylabel("PC2")
ax_gt3dPCA.set_zlabel("PC3")
ax_gt3dPCA.set_title("3D PCA of Ground Truth")
ax_gt3dPCA.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

ax_gt3dPCA.set_xlabel(f"PC1 ({explained_3d[0]*100:.1f}%)")
ax_gt3dPCA.set_ylabel(f"PC2 ({explained_3d[1]*100:.1f}%)")
ax_gt3dPCA.set_zlabel(f"PC3 ({explained_3d[2]*100:.1f}%)")

plt.tight_layout()
# Save the plot.
plt.savefig(os.path.join(baseline_save_directory, "combined_subplots_PCA_k_3.png"), dpi=300)
plt.show()
