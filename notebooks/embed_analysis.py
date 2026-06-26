import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    from bioscience.utils import plotting, table
    import anndata as ad
    import os
    from pyprojroot import here
    import scanpy as sc

    return ad, here, os, plotting, sc


@app.cell
def _():
    import marimo as mo
    import drvi

    return drvi, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load files

        This section loads embedding objects and gene names used by all analyses below.
        It defines the common data sources that feed the tables and plots.
    """)
    return


@app.cell
def _(here, os):
    data_path = os.path.join(here(), "results/L40_S123_260611111342/embed.h5ad")
    return (data_path,)


@app.cell
def _(ad, data_path):
    adata = ad.read_h5ad(data_path)
    return (adata,)


@app.cell
def _(here, os):
    # load gene names file
    with open(os.path.join(here(), "genes.txt"), "r", encoding = "utf-8") as f:
        text = f.read()

    #make it as a list
    genes = [g.strip() for g in text.split()]
    return (genes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Jaccard Index Computation

    This section measures overlap between top-k genes associated with latent dimensions.
    It is useful for identifying potentially redundant latents inside one embedding.

    If two latent dimensions share very similar top-k gene sets, they may encode overlapping biological signals.
    """)
    return


@app.cell
def _(mo):
    topk_choice_jac = mo.ui.number(start = 5, stop = 500, label = "Top-k genes to consider")
    return (topk_choice_jac,)


@app.cell
def _(mo, topk_choice_jac):
    mo.hstack([topk_choice_jac, mo.md(f"Consider top {topk_choice_jac.value} genes")])
    return


@app.cell
def _(adata, genes, mo, plotting, topk_choice_jac):
    mo.center(
            plotting.compute_jaccard(
                embed=adata,
                gene_names=genes, 
                topk=topk_choice_jac.value
            )
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Latent Dimension Statistics

        This section provides distribution-level diagnostics for latent dimensions,
        including spread, magnitude, and reconstruction-related signals.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use this view to quickly detect dominant, collapsed, or low-variance latents.
    """)
    return


@app.cell
def _(adata, mo, plotting):

    mo.center(plotting.plot_latent_dimension_stats(
                    embed=adata,
                    plot_width=480,
                    plot_height=200,
                    grid_columns=2,
                )
            )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Interpretability Plots
    """)
    return


@app.cell
def _(adata, genes, plotting):
    plotting.plot_interpretability_scores(embed=adata, gene_names=genes)
    return


@app.cell
def _(adata, genes, plotting):
    interpretability_score = plotting.get_interpretability_scores(embed=adata, gene_names=genes, hide_vanished=True)
    return (interpretability_score,)


@app.cell
def _(here, interpretability_score, os):
    interpretability_score.to_csv(
        os.path.join(here(), "results/L40_S123_260611111342/interpretability_scores.csv"),
        index = True,
        sep = ";"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gene Scores

    This section shows gene-level score landscapes for each latent dimension.
    It helps identify genes that most strongly define the latent factors
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each point is a gene, positioned by the two interpretability scores.
    Point size encodes combined importance.
    """)
    return


@app.cell
def _(adata, genes, plotting):
    fig = plotting.plot_both_scores(embed=adata,gene_names=genes, hide_vanished=True, ncols = 5)
    fig[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Spatial Plot

        This section projects a selected latent dimension onto spatial coordinates.
        It is useful for visualizing where latent-driven patterns localize across samples.
    """)
    return


@app.cell
def _(adata, mo):
    latent_options = {
        latent_name: int(latent_name.split()[1]) - 1
        for latent_name in sorted(
            adata.var["title"],
            key=lambda x: int(x.split()[1]),
        )
    }
    latent_choice = mo.ui.dropdown(
            options = latent_options,
            value = None,
            label = "Select Latent",
            searchable=True
        )
    return (latent_choice,)


@app.cell
def _(latent_choice):
    latent_choice
    return


@app.cell
def _(adata, latent_choice, mo, plotting):
    mo.stop(latent_choice.value is None)
    spatial_plot = plotting.plot_dr_xy_centroids(
        adata=adata,
        DR=latent_choice.value,
        use_pheno_prefix= False,
        size = 0.1
    )
    spatial_plot.suptitle(f"DR {latent_choice.value + 1}")
    mo.center(spatial_plot)
    return


@app.cell
def _(data_path, os):
    h5ad_name = os.path.splitext(os.path.basename(data_path))[0]
    return (h5ad_name,)


@app.cell
def _(h5ad_name):
    h5ad_name
    return


@app.cell
def _(h5ad_name, os, output_base_folder):
    output_folder = os.path.join(output_base_folder, h5ad_name)
    return (output_folder,)


@app.cell
def _(adata, h5ad_name, os, output_folder):
    # spatial plots
    latent_count = adata.X.shape[1]
    latent_numbers = list(range(latent_count))

    pdf_filename = os.path.join(output_folder, f"{h5ad_name}-spatial_plots.pdf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Latent vs Covariate
    """)
    return


@app.cell
def _(adata):
    adata.obs["run"].unique()
    return


@app.cell
def _(adata, drvi):
    drvi.utils.plotting.plot_latent_dims_in_heatmap(embed = adata, categorical_column = "lv1", title_col = "title", figsize = (20,5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # UMAP
    """)
    return


@app.cell
def _(adata, drvi):
    drvi.utils.pl.plot_latent_dims_in_umap(adata, dim_subset=["DR 1", "DR 2"])
    return


@app.cell
def _(drvi, embed):
    drvi.utils.pl.plot_latent_dims_in_umap(embed, dim_subset=["DR 1", "DR 2"])
    return


@app.cell
def _(adata, sc):
    sc.pl.umap(adata, color = "lv1")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Per Sample Plot
    """)
    return


@app.cell
def _(adata, plotting):
    plotting.plot_sample_all_latents_xy_centroids(adata=adata,sample="110", use_pheno_prefix=False, size = 0.1)
    return


if __name__ == "__main__":
    app.run()
