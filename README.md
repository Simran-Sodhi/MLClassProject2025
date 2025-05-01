# MLClassProject2025

**Machine Learning Methods for Lung Cancer Diagnosis and Subtype Identification from RNA Sequencing Data**
**GitHub Repository:** [MLClassProject2025](https://github.com/Simran-Sodhi/MLClassProject2025)

This repository implements a pipeline for analyzing RNA-seq data from lung cancer samples using various machine learning techniques, including dimensionality reduction, unsupervised clustering, and supervised classification. The goal is to diagnose cancer presence and identify subtypes using RNA sequencing data.

## Overview of Implemented Methods

We implemented and evaluated three main machine learning components:

- **Dimensionality Reduction:** Custom gene filtering pipeline combining variance, MAD, and correlation filtering.
- **Unsupervised Clustering:** K-means clustering using cosine similarity.
- **Supervised Classification:** Random Forest and SVM classifiers for tumor vs normal prediction and subtype classification.

## Repository Structure

We use separate branches for each module of the pipeline, along with a consolidated branch for convenience:

| Branch Name              | Description                                                |
|--------------------------|------------------------------------------------------------|
| `combined_branch`        | Contains the full pipeline, integrating all modules.       |
| `data_preprocessing` | Extract TPM-normalized gene expression matrix from all samples and save as an AnnData object. |
| `dimensionality_reduction` | Code for reducing the number of genes based on variability and redundancy. |
| `batch_effect`           | Code for preprocessing and batch effect correction using pyComBat. |
| `pca_and_clustering`     | PCA visualization and unsupervised clustering via K-means. |
| `classification`         | Random Forest and SVM classifiers for cancer prediction.   |
| `diff_exp_analysis`      | R scripts for differential expression analysis (DESeq2) and GSEA validation. |

## How to Use This Pipeline

1. **Start with a TPM-normalized gene expression matrix** (samples × genes) in the form of an `AnnData` object using code in `data_preprocessing`.
2. **Run the pipeline in this order**:

   - **Dimensionality Reduction:**  
     Use code in `dimensionality_reduction` to filter genes from ~60,000 to ~9,000 using variance, MAD, and correlation filtering.

   - **Batch Correction:**  
     Apply pyComBat from the `batch_effect` branch to remove batch effects across projects.

   - **Unsupervised Clustering:**  
     Use `pca_and_clustering` for K-means clustering to identify potential subtypes.

   - **Supervised Classification:**  
     Use `classification` to:
     - Classify samples as tumor vs. normal.
     - Classify subtypes by using cluster assignments (from K-means) as an additional feature.

   - **Cluster Validation:**  
     - Use raw (uncorrected) count data in the `diff_exp_analysis` branch to perform DESeq2-based DEG analysis and perform GSEA analysis.
