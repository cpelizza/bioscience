import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from dataclasses import asdict, dataclass
    from pyprojroot import here
    import tyro
    import time
    import random
    from datetime import datetime
    from pathlib import Path
    import scvi

    from lightning.pytorch.loggers import WandbLogger

    import torch
    import anndata as ad

    import marimo as mo
    import scanpy as sc

    import drvi


    torch.set_float32_matmul_precision('high')
    from drvi.model import DRVI

    from omegaconf import OmegaConf

    import re

    return DRVI, Path, ad, drvi, here, mo, re, sc


@app.cell
def _(Path):
    ADATA_PATH = Path(
            "/mnt/lustre/scratch/nlsas/home/res/cnag71/resh000982/bioscience/data/full_object_phenotype.h5ad"
        )
    RESULTS_FOLDER = Path(
        "/mnt/netapp1/Store_RES/home/res/cnag71/resh000982/projects/bioscience/results"
    )
    return ADATA_PATH, RESULTS_FOLDER


@app.cell
def _():
    #args = mo.cli_args()
    #folder_name = args.get("folder", "")
    return


@app.cell
def _():
    folder_name = "L10_S456_260612162002"
    return (folder_name,)


@app.cell
def _(ADATA_PATH, Path, RESULTS_FOLDER, folder_name, here, re):
    adata_path = ADATA_PATH

    if not adata_path.is_absolute():
        adata_path = Path(here()) / adata_path
    if not folder_name:
        raise ValueError(
            "No folder provided. Run as:\n"
            "  marimo export html make_embed.py -o report.html -- --folder L20_S0_..."
        )

    io_dir = RESULTS_FOLDER / folder_name
    match = re.match(r"L(\d+)_S(\d+)_", folder_name)
    if not match:
        raise ValueError(
            f"Could not parse latent dim / seed from folder name: {folder_name!r}"
        )
    nlatent = int(match.group(1))
    seed = int(match.group(2))
    model_path = io_dir / f"drvi_model_epochs400_latent{nlatent}_seed{seed}"
    embed_path = io_dir / "embed.h5ad"

    print(f"Folder      : {folder_name}")
    print(f"  adata path: {adata_path}")
    print(f"  model path: {model_path}")
    print(f"  embed path: {embed_path}")
    return adata_path, embed_path, model_path


@app.cell
def _(adata_path, sc):
    print(f"Loading data from {adata_path} ...")
    nhood = sc.read_h5ad(adata_path, backed='r')
    nhood.layers["counts"] = nhood.X
    return (nhood,)


@app.cell
def _(DRVI, ad, model_path, nhood):
    print("Loading model and computing latent representation ...")
    model_trained = DRVI.load(model_path, nhood)

    embed = ad.AnnData(model_trained.get_latent_representation(), obs=nhood.obs)
    return embed, model_trained


@app.cell
def _(embed, embed_path, model_trained, sc):
    print("Setting latent dimension stats ...")
    model_trained.set_latent_dimension_stats(embed, vanished_threshold=0.5)

    print("Calculating gene scores ...")
    model_trained.calculate_interpretability_scores(embed, "OOD")
    model_trained.calculate_interpretability_scores(embed, "IND")

    print("Dimension reduction ...")
    #import rapids_singlecell as rsc
    #rsc.get.anndata_to_GPU(embed)
    sc.pp.neighbors(embed, n_neighbors=10, use_rep="X", n_pcs=embed.X.shape[1])
    sc.tl.umap(embed, spread=1.0, min_dist=0.5, random_state=123)
    sc.pp.pca(embed)
    #sc.get.anndata_to_CPU(embed)

    print(f"Writing embed to {embed_path} ...")
    embed.write_h5ad(embed_path)

    print("Done.")
    return


@app.cell
def _(drvi, embed):
    drvi.utils.pl.plot_latent_dimension_stats(embed, ncols=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Visualize latents in umap
    """)
    return


@app.cell
def _(drvi, embed):
    drvi.utils.pl.plot_latent_dims_in_umap(embed, dim_subset=["DR 1", "DR 2"])
    return


@app.cell
def _(embed, model_trained, nhood):
    model_trained.plot_interpretability_scores(embed, nhood)
    return


if __name__ == "__main__":
    app.run()
