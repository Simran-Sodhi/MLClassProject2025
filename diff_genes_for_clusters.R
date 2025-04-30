# Load metadata file with sample-cluster (and batch) mapping
metadata <- read.csv("DESeq2Input/lung_cluster_Metadata.csv", header=TRUE, row.names=1)

# Make sure clusters are factors
metadata$cluster <- factor(metadata$condition)

# Make sure batch are factors
metadata$batch <- factor(metadata$batch)

# List of unique clusters
unique_clusters <- sort(unique(metadata$cluster))

# Read count data
count_data <- read.csv("DESeq2Input/counts_matrix.csv", header = TRUE, row.names = 1, check.names = FALSE)

# Ensure alignment
if (!all(colnames(count_data) %in% rownames(metadata))) {
  stop("Mismatch between count data columns and metadata row names!")
}

for (cluster_id in unique_clusters) {
  message("Analyzing Cluster: ", cluster_id)
  
  meta_subset <- metadata
  
  # Create one-vs-rest label
  meta_subset$cluster_group <- ifelse(meta_subset$cluster == cluster_id, as.character(cluster_id), "rest")
  meta_subset$cluster_group <- factor(meta_subset$cluster_group, levels = c("rest", as.character(cluster_id)))
  
  if (sum(meta_subset$cluster_group == cluster_id) < 3) {
    message("Skipping cluster ", cluster_id, " — too few samples to analyze.")
    next
  }
  
  # Subset counts to matching samples
  subset_counts <- count_data[, rownames( meta_subset)]
  
  dds <- DESeqDataSetFromMatrix(countData = subset_counts,
                                colData =  meta_subset,
                                design = ~ batch + cluster_group)
  dds <- DESeq(dds)
  res <- results(dds, contrast = c("cluster_group", cluster_id, "rest"))
  res_shrunk <- lfcShrink(dds, coef = 2, type = "apeglm")
  
  # Filter significant genes
  sig_genes <- subset(res_shrunk, padj < 0.05 & abs(log2FoldChange) > 1)
  message("Significant DEGs in cluster ", cluster_id, ": ", nrow(sig_genes))
  
  # Create output folder
  out_dir <- paste0("results/cluster_", cluster_id, "_vs_rest/")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Save results
  write.csv(sig_genes[order(sig_genes$padj, na.last = NA), ],
            file = paste0(out_dir, "sig_genes.csv"))
  write.csv(as.data.frame(res[order(res$padj, na.last = NA), ]),
            file = paste0(out_dir, "all_genes_results.csv"))
}
