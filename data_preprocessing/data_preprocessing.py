# This script extracts tpm_unstranded gene expression values from all tsv files,
# maps them to sample metadata, and compiles the data into an AnnData object for downstream analysis.

import os
import glob
import pandas as pd
from tqdm import tqdm
from collections import Counter
import anndata as ad

# ------------------------------------------------------------------------------
# 1. Read label information
# ------------------------------------------------------------------------------

# Load sample type labels
labels = pd.read_csv('../../labels.csv', sep='\t')
label = labels['SampleType']

# Compute label distribution
counts = Counter(label)
total = sum(counts.values())
proportions = {element: count / total for element, count in counts.items()}

# Print distribution summary
for element, proportion in proportions.items():
    print(f"Element: {element}, Proportion: {proportion:.4f}, Counts: {counts[element]}")

# ------------------------------------------------------------------------------
# 2. Collect gene expression values from TSV files
# ------------------------------------------------------------------------------

# Set the base directory for RNA-seq files
base_dir = "./LungCancer_RNAseq"

# Recursively find all .tsv files under the base directory
tsv_files = glob.glob(os.path.join(base_dir, "**", "*.tsv"), recursive=True)

# Define column to extract from each TSV file
col_names = ['unstranded']

# Iterate through each file and extract the specified expression column
for file_path in tqdm(tsv_files, total=len(tsv_files)):
    for col_name in col_names:
        df = pd.read_csv(file_path, sep="\t", skiprows=[0, 2, 3, 4, 5])

        if "gene_id" not in df.columns or col_name not in df.columns:
            print(f"Skipping {file_path} because it lacks 'gene_id' or '{col_name}' column.")
            continue

        # Extract the values of the desired column
        values = df[col_name].tolist()

        # Convert list of values to comma-separated string
        output_str = ", ".join(map(str, values))

        # Append expression values to output file
        with open(f'./{col_name}.txt', mode='a') as f:
            f.write(output_str + "\n")

# ------------------------------------------------------------------------------
# 3. Create AnnData object for downstream analysis
# ------------------------------------------------------------------------------

# Gene metadata
genes = pd.read_csv('project/gene_list.csv', sep='\t')

# Extract sample names from file paths
obs_names = [s.split('/')[2] for s in tsv_files]

# Construct AnnData for each specified column
for col in col_names:
    # Read expression matrix (each row corresponds to a sample)
    df = pd.read_csv(f'./{col}.txt', header=None, sep=', ')
    print(f"Expression matrix shape: {df.shape}")

    # Re-read label file for mapping
    sample_type = []

    # Map file names to sample types
    for obs in obs_names:
        match = labels[labels.iloc[:, 0] == obs]
        if not match.empty:
            sample_type.append(match.iloc[0, 1])
        else:
            sample_type.append("Unknown")  # fallback for unmatched entries

    print(f"Number of sample types mapped: {len(sample_type)}")

    # Create observation metadata
    obs_meta = pd.DataFrame({'sample_type': sample_type}, index=obs_names)

    # Store into AnnData format
    adata = ad.AnnData(X=df.values, obs=obs_meta, var=genes)

    # Save AnnData object
    adata.write(f'./{col}.h5ad', compression="gzip")
