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

    return (
        DRVI,
        OmegaConf,
        Path,
        WandbLogger,
        ad,
        asdict,
        dataclass,
        datetime,
        drvi,
        here,
        mo,
        os,
        random,
        sc,
        scvi,
        time,
        tyro,
    )


@app.cell
def _(dataclass, here, os, tyro):
    @dataclass
    class DRVIConfig:
        n_latent: int = 10
        n_epochs: int = 400
        seed: int = 456
        batch_key: str = "batch"
        early_stopping: bool = False
        early_stopping_patience: int = 20
        adata: str = "/mnt/lustre/scratch/nlsas/home/res/cnag71/resh000982/bioscience/data/full_object_phenotype.h5ad"
        results_dir: str = os.path.join(here(), "results")
        startup_jitter_seconds: float = 5.0
        object: str = "undefined"

    config = tyro.cli(DRVIConfig)
    config
    return (config,)


@app.cell
def _(Path, config, datetime, here, os, random, scvi, time):
    # Jitter to prevent filesystem race conditions on simultaneous array jobs
    if config.startup_jitter_seconds > 0:
        time.sleep(config.startup_jitter_seconds + random.uniform(0, 5))

    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    random.seed(config.seed)
    scvi.settings.seed = config.seed

    results_root = Path(config.results_dir)
    if not results_root.is_absolute():
        results_root = Path(here()) / results_root
    results_path = results_root / f"L{config.n_latent}_S{config.seed}_{timestamp}"
    os.makedirs(results_path, exist_ok=True)
    return results_path, timestamp


@app.cell
def _(WandbLogger, asdict, config, results_path, timestamp):
    wand_logger = WandbLogger(
        project="bioscience_lab",
        name=timestamp,
        save_dir=str(results_path),
        log_model=False,
        config=asdict(config),
    )
    return (wand_logger,)


@app.cell
def _(Path, config, here, sc):
    adata_path = Path(config.adata)
    if not adata_path.is_absolute():
        adata_path = Path(here()) / adata_path
    print(f"Loading data from {adata_path} ...")
    nhood = sc.read_h5ad(adata_path, backed='r')
    nhood.layers["counts"] = nhood.X
    return (nhood,)


@app.cell
def _(DRVI, config, nhood):
    DRVI.setup_anndata(
            nhood,
            layer="raw_counts",
            batch_key=config.batch_key,
            is_count_data=True,
        )
    return


@app.cell
def _(DRVI, config, nhood, scvi):
    scvi.settings.seed = config.seed
    model = DRVI(
        nhood,
        n_latent=config.n_latent,
        encoder_dims=[config.n_latent, config.n_latent],
        decoder_dims=[config.n_latent, config.n_latent],
    )
    return (model,)


@app.cell
def _(config, io_dir, model, wand_logger):
    model_path = io_dir / f"drvi_model_epochs{config.n_epochs}_latent{config.n_latent}_seed{config.seed}"
    print(f"Training: n_latent={config.n_latent}, seed={config.seed}, epochs={config.n_epochs}")
    model.train(
        max_epochs=config.n_epochs,
        early_stopping=config.early_stopping,
        early_stopping_patience=config.early_stopping_patience,
        plan_kwargs={"n_epochs_kl_warmup": config.n_epochs},
        logger=wand_logger,
    )
    model.save(model_path, overwrite=True)
    return (model_path,)


@app.cell
def _(OmegaConf, asdict, config, io_dir):
    _conf_obj = OmegaConf.create(asdict(config))
    OmegaConf.save(config=_conf_obj, f=io_dir / "config.yaml")
    return


@app.cell
def _(DRVI, ad, io_dir, model_path, nhood, sc):
    model_trained = DRVI.load(model_path, nhood)
    embed = ad.AnnData(model_trained.get_latent_representation(), obs=nhood.obs)

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
    embed_path = io_dir / "embed.h5ad"
    print(f"Writing embed to {embed_path} ...")
    embed.write_h5ad(embed_path)

    print("Done.")
    return (embed,)


@app.cell
def _(drvi, embed):
    drvi.utils.pl.plot_latent_dimension_stats(embed, ncols=2)
    return


@app.cell
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
def _(embed, model, nhood):
    model.plot_interpretability_scores(embed, nhood)
    return


if __name__ == "__main__":
    app.run()
