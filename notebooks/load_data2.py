import pandas as pd
import numpy as np
from pydantic_core import Some

import anndata as ad
import scanpy as sc

import os

# The datasets contains data obtained from colorectal tissue samples. 
# In practice, a biological tissue sample is cut into very thin slices, called tissue sections. These sections are then analyzed using a technology 
# called CosMx SMI, which measures the expression of many genes directly inside the tissue while preserving the spatial position of the cells.

# This is important because, in standard single-cell experiments, cells are often separated from the tissue before being analyzed, so their original 
# position is lost. In spatial transcriptomics, instead, we can know both which genes are active in each cell and where that cell is located in the tissue. 
# This allows us to study not only which cell types are present, but also how they are organized in space and how they relate to different tissue regions.

# The data are provided as .h5ad files. This is a common file format used in Python for single-cell and spatial transcriptomics analysis. 
# Each .h5ad file corresponds to one analyzed tissue section. Inside each file there is the gene expression matrix, meaning the measured expression 
# levels of genes in each cell, together with cell metadata, such as cell type annotations, tissue region annotations, and spatial or biological labels.

# The file identifiers, such as 110, 210, or 242, should not be interpreted directly as patient IDs. 
# According to the Zenodo page linked to the H&E stain data, the identifiers describe:

# run + slide + section
# This means that the number gives technical and sample-organization information. For example, an identifier like 221 can be read as:
# 2 = experimental run
# 2 = slide
# 1 = section on that slide
# So the identifier tells us from which run, slide, and tissue section the data come. It does not automatically mean “patient 221”.

# A simplified interpretation is:
# donor / patient
#  → tissue sample
#    → slide
#      → tissue section
#        → cells measured by CosMx SMI

# Some slides contain only one tissue section, while others contain more than one section. The Zenodo metadata also indicate that most sections come 
# from unique donors, but not all section identifiers should be treated as completely independent patients. Therefore, in the project it is more accurate 
# to describe the .h5ad files as tissue sections or samples/sections, rather than assuming that each file corresponds to a different patient.

# In summary, the dataset can be understood as:
#multiple donors
#  → colorectal tissue samples
#    → thin tissue sections
#      → individual cells measured in space
#        → gene expression + spatial position + biological annotations

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
# The goal is to:
# 1. Load multiple tissue sections
# 2. Perform QC filtering
# 3. Merge them into one dataset
# 4. Normalize and preprocess for downstream analysis (e.g., DRVI)

slide_paths = [
    ("110", "data/110.h5ad"),
    ("210", "data/210.h5ad"),
    ("221", "data/221.h5ad"),
    ("222", "data/222.h5ad"),
    ("231", "data/231.h5ad"),
    ("232", "data/232.h5ad"),
    ("242", "data/242.h5ad"),
]

adatas = []

# ------------------------------------------------------------------
# LOAD + QC FILTER
# ------------------------------------------------------------------


for batch_id, path in slide_paths:
    print(f"Loading {batch_id}")

     # Load AnnData in backed mode (memory-efficient read-only access)
    adata = sc.read_h5ad(
        os.path.join(os.getcwd(), path), backed="r"
    )

    # ------------------------------------------------------------------
    # QUALITY CONTROL FILTERING
    # ------------------------------------------------------------------

    # Case 1: QC filtering column exists
    # Keep only QC-passed cells if available
    if "fil" in adata.obs.columns:
        before = adata.n_obs

        mask_cells = adata.obs["fil"].values
        mask_genes = ~adata.var_names.str.contains(
        "Negative|Control|System",
        case=False,
        regex=True,
    )
        # Apply both cell and gene filters
        adata_final = adata[mask_cells, mask_genes]

        after = adata_final.n_obs

        print(
            f"  QC filter and probe removal: {before:,} -> {after:,} cells"
        )

    # Case 2: No QC column available
    else:
        before = adata.n_obs
        
        # Only perform gene filtering
        mask_genes = ~adata.var_names.str.contains(
        "Negative|Control|System",
        case=False,
        regex=True,
    )
        adata_final = adata[:, mask_genes]

        after = adata_final.n_obs

        print(
            f"  Probe removal: {before:,} -> {after:,} cells"
        )
    # Store processed section
    adatas.append(adata_final)

# ------------------------------------------------------------------
# MERGE ALL TISSUE SECTIONS
# ------------------------------------------------------------------
print("Merging sections...")
phenotype_merged = ad.concat(
    adatas,
    axis=0,                 # concatenate cells (observations)
    join="inner",          # keep only shared genes across batches
    label="batch",         # create batch annotation column
    keys=[batch for batch, _ in slide_paths],  # batch IDs
    index_unique="-",      # ensure unique cell IDs
)

# ------------------------------------------------------------------
# CELL TYPE DISTRIBUTION (OPTIONAL QC CHECK)
# ------------------------------------------------------------------

if "lv2" in phenotype_merged.obs.columns:
    phenotype_lv2_counts = (
        phenotype_merged.obs["lv2"]
        .value_counts()
    )

    print(phenotype_lv2_counts)

# ------------------------------------------------------------------
# SAVE RAW COUNTS BEFORE NORMALIZATION FOR DRVI
# ------------------------------------------------------------------
print("Saving raw counts in .layers['raw_counts']...")

# Store original count matrix for downstream models (e.g., DRVI)
phenotype_merged.layers["raw_counts"] = phenotype_merged.X.copy()

# ------------------------------------------------------------------
# TOTAL COUNTS NORMALIZATION
# ------------------------------------------------------------------
print("Normalizing total counts to 10,000...")
sc.pp.normalize_total(
    phenotype_merged,
    target_sum=1e4,
)

# ------------------------------------------------------------------
# LOG TRANSFORM
# ------------------------------------------------------------------
print("Taking log1p transform...")
sc.pp.log1p(phenotype_merged)

# ------------------------------------------------------------------
# HIGHLY VARIABLE GENE (HVG) SELECTION
# ------------------------------------------------------------------
print("Selecting highly variable genes (HVGs) with batch-aware method...")
sc.pp.highly_variable_genes(
    phenotype_merged,
    n_top_genes=2000,
    batch_key="batch",   # ensures HVGs are consistent across batches
    subset=True,         # keep only HVGs
)


print(
    f"Selected HVGs: {phenotype_merged.n_vars:,}"
)

# ------------------------------------------------------------------
# SAVE PROCESSED OBJECT
# ------------------------------------------------------------------
print("Saving processed object to .h5ad...")
output_path = os.path.join(
    os.getcwd(),
    "preprocessing/full_object_phenotype.h5ad",
)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Write final AnnData object
phenotype_merged.write_h5ad(output_path)

print(f"Saved to:\n{output_path}")