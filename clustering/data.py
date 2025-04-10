import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from K_means import spherical_kmeans

def normalize_rows(A):
    """Row‑normalize A to unit ℓ2 norm."""
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return A / norms

def compute_avg_inertia(X, labels, centroids):
    """
    Average within‑cluster squared cosine‑distance:
      (1/n) ∑_i (1 − cos(x_i, c_{z_i}))^2
    """
    Xn = normalize_rows(X)
    Cn = normalize_rows(centroids)
    dots = Xn.dot(Cn.T)  # (n_samples, k)
    assigned = dots[np.arange(X.shape[0]), labels]
    dists = 1.0 - assigned
    return np.mean(dists**2)

def elbow_plot(X, ks, max_iter=100):
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
    plt.savefig('elbow_plot.png')
    plt.show()
    return inertias

# --- Load & QC ---
adata = sc.read_h5ad('data/tpm_unstranded.h5ad')
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=10)
X = adata.X  # raw matrix (cells × genes)

X = sklearn.preprocessing.StandardScaler().fit_transform(X)

# --- Elbow analysis ---
ks = list(range(1, 11))
elbow_plot(X, ks, max_iter=100)

# --- Choose k (e.g. from elbow) ---
k = 2
labels, centroids = spherical_kmeans(X, k, max_iter=100)

# --- PCA visualization ---
pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
centroids_pca = pca.transform(centroids)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centroids_pca[:,0], centroids_pca[:,1],
            c='red', marker='x', s=100, label='Centroids')
plt.title("PCA of Spherical k‑Means (k=%d)" % k)
plt.legend()
plt.savefig('kmeans_pca.png')
plt.show()

# --- tSNE visualization ---
combined = np.vstack([X, centroids])
tsne = TSNE(n_components=2, perplexity = 50, random_state=42)
combined_tsne = tsne.fit_transform(combined)
X_tsne = combined_tsne[:X.shape[0]]
centroids_tsne = combined_tsne[X.shape[0]:]

plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centroids_tsne[:,0], centroids_tsne[:,1],
            c='red', marker='x', s=100, label='Centroids')
plt.title("tSNE of Spherical k‑Means (k=%d)" % k)
plt.legend()
plt.savefig('kmeans_tsne.png')
plt.show()

# --- Ground truth labeling ---
def classify_sample(sample):
    if "Normal" in sample:
        return "Normal"
    elif sample == "Not Reported":
        return np.nan
    else:
        return "Cancer"

gt = pd.read_csv('data/labels.csv', sep='\t')
gt['Class'] = gt['SampleType'].apply(classify_sample)
adata.obs = adata.obs.merge(gt, left_index=True, right_on='FileID', how='left')
adata.obs['FileID'] = adata.obs['FileID'].astype(str)
adata.obs.index = adata.obs['FileID'].str[:36]
adata.obs['Class'] = adata.obs['Class'].fillna("Not Reported")

true_labels = adata.obs['Class']
le = LabelEncoder()
true_labels_enc = le.fit_transform(true_labels)
unique_labels = le.classes_
colors = plt.cm.viridis(np.linspace(0,1,len(unique_labels)))
patches = [mpatches.Patch(color=colors[i], label=unique_labels[i])
           for i in range(len(unique_labels))]

# --- PCA with ground truth ---
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=true_labels_enc,
            cmap='viridis', alpha=0.6)
plt.title("PCA of Ground Truth")
plt.legend(handles=patches, bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig('gt_pca.png')
plt.show()

# --- tSNE with ground truth ---
plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=true_labels_enc,
            cmap='viridis', alpha=0.6)
plt.title("tSNE of Ground Truth")
plt.legend(handles=patches, bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig('gt_tsne.png')
plt.show()
