# BIOSCIENCE PROJECT

This repository contains the code, notebooks, and outputs developed for a university examination project on **unsupervised DRVI decomposition of a CosMx WTx spatial atlas into literature-validated gene programmes**.

The analysis integrates spatial transcriptomics data with latent representation learning in order to identify biologically meaningful gene programmes and assess their spatial organization and interpretability.

## Project objective

The central aim of the project is to:
- learn latent factors from a CosMx spatial transcriptomics atlas using **DRVI**,
- interpret the learned dimensions in terms of gene programmes,
- evaluate redundancy and biological coherence across latent factors,
- and validate the resulting patterns against the scientific literature.

## Reproducibility and environment

The project is designed to be executed using the **Pixi** environment defined in [`pixi.toml`](pixi.toml).

To reproduce the analysis, install Pixi and run:

```bash
pixi install
pixi shell
```

## Repository Structure

### notebooks/
Contains the main analysis notebooks and scripts used throughout the project.

load_data2.py — loads the CosMx .h5ad tissue-section files, performs preprocessing, and prepares the merged dataset for downstream analysis.

drvi_train.py — trains the DRVI model on the prepared AnnData object and stores run-specific outputs.

generate_analysis.py — loads trained models and generates the latent embedding used for downstream interpretation.

embed_analysis.py — performs exploratory analysis of the embedding, including latent statistics, Jaccard overlap of top genes, and interpretability diagnostics.

generate_plots.py — produces PDF reports and spatial visualizations of latent dimensions, both globally and by sample.

extract_gene_list.py — exports the gene identifiers used in the analysis to a text file.

create_embeds_manual.py and create_embeds_manual_copy.py — manual variants of the embedding-generation workflow used for specific trained runs.

### src/
Contains the Python package code used by the notebooks and scripts.

src/bioscience/utils/plotting.py — plotting utilities for correlation heatmaps, Jaccard similarity, latent statistics, interpretability plots, and related visual summaries.

src/bioscience/utils/table.py — helper functions for tabular summaries, including Moran’s I spatial autocorrelation computations and derived summary tables.

## results/
Contains outputs from DRVI training and downstream analyses.

Typical contents of a result folder include:

config.yaml — model and training configuration for a specific run,

embed.h5ad — latent embedding produced by the trained DRVI model,

model checkpoints or serialized model artifacts,
gene lists and derived outputs used for interpretation and visualization.

### genes.txt
A list of gene symbols used for downstream interpretation and figure generation.

### pixi.toml
Defines the reproducible computational environment required to execute the notebooks and scripts in this repository.

### pixi.lock
Lockfile for the Pixi environment, ensuring reproducible dependency resolution.

### pyproject.toml
Project metadata and Python package configuration.

## Data description
The analysis is based on CosMx WTx spatial transcriptomics data (https://zenodo.org/records/15574384) stored in .h5ad format. Each file represents a spatially resolved tissue section containing:

gene expression measurements,
cell-level annotations,
batch or sample metadata,
and spatial coordinates.
