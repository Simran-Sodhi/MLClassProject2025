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
import matplotlib.gridspec as gridspec
import os

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
    plt.savefig('./output/elbow_plot.png')
    plt.show()
    return inertias

# --- Load & QC ---
adata = sc.read_h5ad('./tpm_unstranded_subset_corrected.h5ad')
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=10)
X = adata.X  # raw matrix (cells × genes)

X = sklearn.preprocessing.StandardScaler().fit_transform(X)

# # --- Elbow analysis ---
ks = list(range(1, 11))
elbow_plot(X, ks, max_iter=100)

# --- Choose k (e.g. from elbow) ---
k = 3
labels, centroids = spherical_kmeans(X, k, max_iter=100)

# --- PCA visualization ---
# 2D-PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
centroids_pca = pca.transform(centroids)

# # 3D-PCA
pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X)
print(pca_3d.explained_variance_ratio_)
centroids_pca_3d = pca_3d.transform(centroids)


# --- tSNE visualization ---
# 2D-tSNE
combined = np.vstack([X, centroids])
tsne = TSNE(n_components=2, perplexity = 50, random_state=42)
combined_tsne = tsne.fit_transform(combined)
X_tsne = combined_tsne[:X.shape[0]]
centroids_tsne = combined_tsne[X.shape[0]:]


# 3D-tSNE
combined = np.vstack([X, centroids])
tsne_3d = TSNE(n_components=3, perplexity=50, random_state=42)
combined_tsne_3d = tsne_3d.fit_transform(combined)
X_tsne_3d = combined_tsne_3d[:X.shape[0]]
centroids_tsne_3d = combined_tsne_3d[X.shape[0]:]


# --- Ground truth labeling ---
def classify_sample(sample):
    if "Normal" in sample:
        return "Normal"
    if "Metastatic" in sample:
        return "Metastatic"
    elif sample == "Not Reported":
        return np.nan
    else:
        return "Cancer"


true_labels = adata.obs['Sample Type']
true_labels = true_labels.map(classify_sample)

le = LabelEncoder()
true_labels_enc = le.fit_transform(true_labels)
unique_labels = le.classes_
colors = plt.cm.viridis(np.linspace(0,1,len(unique_labels)))
patches = [mpatches.Patch(color=colors[i], label=unique_labels[i])
           for i in range(len(unique_labels))]


baseline_save_directory = './output'
os.makedirs(baseline_save_directory, exist_ok=True)
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

# --- Assume the following variables are already computed ---
# X_pca, centroids_pca            : 2D PCA for k‑Means (cells×2, clusters×2)
# X_pca_3d, centroids_pca_3d      : 3D PCA for k‑Means (cells×3, clusters×3)
# X_tsne, centroids_tsne          : 2D tSNE for k‑Means (cells×?, use columns 0 & 1)
# X_tsne_3d, centroids_tsne_3d    : 3D tSNE for k‑Means (cells×3, clusters×3)
# labels                          : cluster labels from spherical k‑Means (for k‑Means plots)
# true_labels_enc                 : numeric labels for ground truth
# Also assume that the corresponding plots for ground truth (PCA/tSNE) use the same embeddings
#   but color the points by true_labels_enc instead.
explained = pca.explained_variance_ratio_
explained_3d = pca_3d.explained_variance_ratio_

# Create a figure with a grid of PCA results.
fig = plt.figure(figsize=(20, 20))
# Create a gridspec with 2 rows, 4 columns.
gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)

##########################
# --- Row 1: Spherical k-Means Results
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
# Add legend patches if desired (using your pre‐created patches variable)
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
plt.savefig(os.path.join(baseline_save_directory, "combined_subplots_PCA_k_2_label_3.png"), dpi=300)
plt.show()


# Create a result for t-SNE
# Create a figure with a grid of PCA results.
fig = plt.figure(figsize=(20, 20))
# Create a gridspec with 2 rows, 4 columns.
gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)

##########################
# --- Row 1: Spherical k-Means Results
##########################

# (0,0): 2D tSNE of k-Means
ax_2dtSNE = fig.add_subplot(gs[0, 0])
ax_2dtSNE.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.6)
ax_2dtSNE.scatter(centroids_tsne[:, 0], centroids_tsne[:, 1], c='red', marker='x', s=100, label='Centroids')
ax_2dtSNE.set_title("2D tSNE of Spherical k‑Means (k=%d)" % k)
ax_2dtSNE.legend()

# (0,1): 3D tSNE of k-Means
ax_3dtSNE = fig.add_subplot(gs[0, 1], projection='3d')
ax_3dtSNE.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2],
                 c=labels, cmap='viridis', s=50, alpha=0.6)
ax_3dtSNE.scatter(centroids_tsne_3d[:, 0], centroids_tsne_3d[:, 1], centroids_tsne_3d[:, 2],
                 c='red', marker='x', s=100, label='Centroids')
ax_3dtSNE.set_xlabel("tSNE1")
ax_3dtSNE.set_ylabel("tSNE2")
ax_3dtSNE.set_zlabel("tSNE3")
ax_3dtSNE.set_title("3D tSNE of Spherical k‑Means (k=%d)" % k)
ax_3dtSNE.legend()

##########################
# --- Row 2: Ground Truth Plots
##########################

# (1,0): 2D tSNE with Ground Truth
ax_gt2dtSNE = fig.add_subplot(gs[1, 0])
ax_gt2dtSNE.scatter(X_tsne[:, 0], X_tsne[:, 1], c=true_labels_enc, cmap='viridis', alpha=0.6)
ax_gt2dtSNE.set_title("2D tSNE of Ground Truth")
ax_gt2dtSNE.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

# (1,1): 3D tSNE with Ground Truth
ax_gt3dtSNE = fig.add_subplot(gs[1, 1], projection='3d')
ax_gt3dtSNE.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2],
                   c=true_labels_enc, cmap='viridis', s=50, alpha=0.6)
ax_gt3dtSNE.set_xlabel("tSNE1")
ax_gt3dtSNE.set_ylabel("tSNE2")
ax_gt3dtSNE.set_zlabel("tSNE3")
ax_gt3dtSNE.set_title("3D tSNE of Ground Truth")
ax_gt3dtSNE.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(baseline_save_directory, "combined_subplots_t-SNE_k_2_label_3.png"), dpi=300)
plt.show()

