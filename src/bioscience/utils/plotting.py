
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from plottable.plots import circled_image
from plottable.cmap import normed_cmap
import matplotlib.cm as cm



import altair as alt

import plotly.express as px

from typing import Sequence, Literal

def make_corr_and_plot(embed1, embed2):

    """
    Compute and plot correlations between latent dimensions from two embeddings.
    
    This function computes the pairwise Pearson correlation between every latent
    dimension in `embed1` and every latent dimension in `embed2`. The resulting
    correlation matrix is reordered so that dimensions with the strongest absolute
    correlations appear first. A heatmap of the absolute, sorted correlations is
    then displayed.
    
    Parameters
    ----------
    embed1
        AnnData-like object containing the first set of latent dimensions in `.X`.
        Columns of `.X` are interpreted as latent dimensions.
    
    embed2
        AnnData-like object containing the second set of latent dimensions in `.X`.
        Columns of `.X` are interpreted as latent dimensions.
    
    Returns
    -------
    numpy.ndarray
        Pairwise Pearson correlation matrix with shape
        `(embed1.X.shape[1], embed2.X.shape[1])`. Entry `(i, j)` contains the signed
        Pearson correlation between latent dimension `i` from `embed1` and latent
        dimension `j` from `embed2`.
    
    Notes
    -----
    The displayed heatmap uses the absolute values of the correlations after
    sorting rows and columns by their strongest absolute correlation. However, the
    returned matrix contains the original signed correlations and is not sorted.
    
    The function expects `embed1.X` and `embed2.X` to have the same number of rows,
    corresponding to matched observations or cells.
    
    Examples
    --------
    >>> cors = make_corr_and_plot(embed1, embed2)
    
    >>> # Inspect the correlation between the first latent dimension of each embedding
    >>> cors[0, 0]
    
    >>> # Find the strongest match in embed2 for each dimension in embed1
    >>> strongest_matches = np.argmax(np.abs(cors), axis=1)
    """

    latents1 = embed1.X
    latents2 = embed2.X
    
    cors = np.zeros((latents1.shape[1], latents2.shape[1]))
    for i in range(latents1.shape[1]):
        for j in range(latents2.shape[1]):
            cors[i,j] = scipy.stats.pearsonr(latents1[:,i], latents2[:,j])[0]

    # strongest absolute correlation for each A column
    row_strength = np.max(np.abs(cors), axis=1)

    # sort A columns by strongest match, descending
    row_order = np.argsort(-row_strength)

    # for B columns, sort by strongest match to sorted A columns
    col_strength = np.max(np.abs(cors[row_order, :]), axis=0)
    col_order = np.argsort(-col_strength)

    C_sorted = np.abs(cors[row_order][:, col_order])

    sns.heatmap(C_sorted, cmap = "Blues")
    plt.show()
    return cors

def compute_jaccard(embed, gene_names, topk):
    """
    Compute and plot the Jaccard similarity between top genes of latent dimensions.
    
    This function computes interpretability scores for each latent dimension, ranks
    genes by their scores, and selects the top `topk` genes for each dimension. It
    then computes the pairwise Jaccard similarity between the selected gene sets of
    all latent dimensions and displays the resulting similarity matrix as a heatmap.
    
    Parameters
    ----------
    embed
        AnnData object containing latent dimensions and interpretability information.
    
    gene_names
        Gene annotation or list of gene names used by `get_interpretability_scores`
        to label the computed scores.
    
    topk
        Number of top-scoring genes to select for each latent dimension before
        computing Jaccard similarity.
    
    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure containing the Jaccard similarity heatmap.
    
    Notes
    -----
    The Jaccard similarity between two latent dimensions is computed as the size of
    the intersection of their top gene sets divided by the size of their union.
    
    The resulting Jaccard matrix is symmetric, with values between 0 and 1. A value
    of 1 means that two latent dimensions have identical top gene sets, while a value
    of 0 means that they share no top genes.
    
    Examples
    --------
    >>> compute_jaccard(embed, gene_names, topk=20)
    
    >>> # Compare latent dimensions using their top 50 genes
    >>> compute_jaccard(
            embed,
            gene_names=gene_names,
            topk=50,
        )
    """
    scores = get_interpretability_scores(embed, gene_names) 

    # rank gene basing on their score
    ranked_genes = {
        latent: scores[latent].sort_values(ascending=False).head(topk).index.tolist()
        for latent in scores.columns
    }

    latents = list(ranked_genes.keys())

    # convert lists → sets once (important for speed)
    gene_sets = {k: set(v) for k, v in ranked_genes.items()}

    jaccard = pd.DataFrame(index=latents, columns=latents, dtype=float)

    for i, a in enumerate(latents):
        A = gene_sets[a]
        for j, b in enumerate(latents):
            if j < i:
                continue

            B = gene_sets[b]

            inter = len(A & B)
            union = len(A | B)

            score = inter / union if union else 0.0

            jaccard.loc[a, b] = score
            jaccard.loc[b, a] = score  # symmetry

    fig, ax = plt.subplots()
    sns.heatmap(jaccard, cmap="Blues", ax=ax)
    return fig

def plot_latent_dimension_stats(embed,
                                columns: Sequence[str] = ("reconstruction_effect", "max_value", "mean", "std"),
                                 titles: dict[str, str] | None = None,
                                log_scale: bool | Literal["try"] = "try",
                               remove_vanished: bool = False,
                                plot_width: int = 600,
                                plot_height: int = 220,
                                grid_columns: int = 2):
    
    """
    Plot the statistics of latent dimensions.

    This function creates line plots showing selected statistics of latent
    dimensions across their ranking order. Each statistic is plotted in a fixed-size
    Altair subplot, and the plots are arranged in a grid. Points are colored by
    whether the corresponding latent dimension is vanished, and dimensions can be
    selected interactively by clicking points.

    Parameters
    ----------
    embed
        Annotated data object containing the latent dimensions and their statistics
        in the `.var` attribute.

    columns
        The columns from `embed.var` to plot. These should be numeric columns
        containing dimension statistics.

    titles
        Custom titles for each column in the plot. If None, default titles are used.

    log_scale
        Whether to use a log scale for the y-axis. If "try", log scale is used
        only if the minimum value is greater than 0.

    remove_vanished
        Whether to exclude vanished dimensions from the plot

    plot_width
        Width of each individual Altair subplot in pixels.

    plot_height
        Height of each individual Altair subplot in pixels.

    grid_columns
        Number of columns to use in the final plot grid.


    Returns
    -------
    altair.vegalite.v6.api.ConcatChart
        An Altair concatenated chart containing one fixed-size subplot for each selected statistic.

    Notes
    -----
    The function expects the following columns in `embed.var`:
    - `order`: Ranking of dimensions
    - `vanished`: Boolean indicating vanished dimensions
    - The columns specified in the `columns` parameter

    When `log_scale="try"`, columns containing zero or negative values are plotted
    with a linear y-axis.

    The interactive point selection is shared across plots through the `title`
    field, so selecting a dimension highlights the corresponding point in each
    subplot.

    Note: in the current implementation, selection may need to be initiated from subplots after the first one.
    
    Examples
    --------
    # Default plot
    >>> plot_latent_dimension_stats(embed)
    
    >>> # Plot only selected statistics
    >>> plot_latent_dimension_stats(
     embed,
     columns=["reconstruction_effect", "max_value"],
    )
    
    >>> # Plot with custom titles and fixed subplot dimensions
    >>> titles = {
     "reconstruction_effect": "Reconstruction Impact",
     "max_value": "Max Activation",
    }
    >>> plot_latent_dimension_stats(
     embed,
     columns=["reconstruction_effect", "max_value"],
     titles=titles,
     log_scale=True,
     plot_width=350,
     plot_height=250,
     grid_columns=2,
    )

    >>> # Exclude vanished dimensions
    >>> plot_latent_dimension_stats(embed, remove_vanished=True)
    
    """


    if titles is None:
        titles = {
            "reconstruction_effect": "Reconstruction effect",
            "max_value": "Max value",
            "mean": "Mean",
            "std": "Standard Deviation",
        }


    df = embed.var.copy()
    df = df.sort_values("order")

    if remove_vanished:
            df = df.query("vanished == False")

    
    selector = alt.selection_point(
                fields=["title"],   # better than encodings=["x"]
                on="click",
                empty="none",
                clear="dblclick"
            )

    plots = []
    for col in columns:
        use_log = False

        if isinstance(log_scale, str):
            if log_scale == "try":
                use_log = df[col].min() > 0
        else:
            use_log = bool(log_scale)

        y_scale = alt.Scale(type="log") if use_log else alt.Scale(type="linear")

                

        base = alt.Chart(
            df,
            title=titles[col],
            width=plot_width,
            height=plot_height,
        ).encode(
            x=alt.X("title:N", sort=df["title"].tolist()),
            y=alt.Y(f"{col}:Q", scale=y_scale),
            tooltip=["title", "vanished"],
        )

        vanished_color = alt.Color(
            "vanished:N",
            title="Vanished",
            scale=alt.Scale(
                domain=[False, True],
                range=["blue", "gray"],
            ),
        )
    
        graph = (
            base.mark_line(color="black")
            +
            base.mark_point(filled=True).encode(
                color=vanished_color,
                size=alt.condition(
                    selector,
                    alt.value(100),
                    alt.value(50),
                ),
                stroke=alt.condition(
                    selector,
                    alt.value("black"),
                    alt.value(None),
                ),
                strokeWidth=alt.condition(
                    selector,
                    alt.value(2),
                    alt.value(0),
                ),
            )
        )
    
        plots.append(graph)


    return alt.concat(
        *plots,
        columns=grid_columns,
        spacing=20,
    ).add_params(selector)


def plot_both_scores(embed,
                     gene_names,
                     ncols = 5,
                     score_threshold = 0.1,
                     hide_vanished = True,
                     dim_subset: Sequence[str] | None = None,
                    ):

    """
    Plot minimum and maximum interpretability scores for latent dimensions.

    This function computes interpretability scores for each latent dimension using
    three score definitions: minimum possible out-of-distribution score, maximum
    possible out-of-distribution score, and combined out-of-distribution score.
    For each selected latent dimension, it creates a faceted scatter plot where
    each point represents a gene. The x-axis shows the minimum score, the y-axis
    shows the maximum score, and the point size represents the combined score.
    
    Only latent dimensions whose maximum combined score is greater than or equal to
    `score_threshold` are plotted. Optionally, the plot can be restricted to a
    subset of latent dimensions and vanished dimensions can be hidden.
    
    Parameters
    ----------
    embed
        AnnData object containing interpretability scores.
        
    ncols
        Number of columns to use in the subplot grid.
    
    gene_names
        Gene annotation or list of gene names used to label the interpretability scores.
    
    score_threshold
        Minimum score threshold for dimensions to be plotted.
    
    hide_vanished
        Whether to exclude vanished latent dimensions from the score computation and plot.
    
    dim_subset
        Optional list of dimension titles to plot. If None, all dimensions
        meeting the threshold are plotted.

    Returns
    -------
    plotly.graph_objects.Figure
        Faceted scatter plot showing minimum and maximum interpretability scores for
        each selected latent dimension.

    pandas.DataFrame
        Long-format DataFrame containing the minimum, maximum, and combined
        interpretability scores for each plotted gene and latent dimension.


    Notes
    -----
    The function expects `get_interpretability_scores` to return a DataFrame where
    rows correspond to genes and columns correspond to latent dimensions.
    
    The plotted facets are ordered according to the latent dimension order returned
    by the interpretability score DataFrames.
     
    Examples
    --------
    >>> plot_both_scores(embed)
    
    >>> # Plot only dimensions with combined score at least 0.2
    >>> plot_both_scores(
            embed,
            score_threshold=0.2,
        )
    
    >>> # Plot a selected subset of latent dimensions
    >>> plot_both_scores(
            embed,
            dim_subset=["latent_1", "latent_5", "latent_12"],
        )
    
    >>> # Include vanished dimensions in the score computation
    >>> plot_both_scores(
            embed,
            hide_vanished=False,
        )
        
    """

    min_df = get_interpretability_scores(embed, gene_names= gene_names, key = "OOD_min_possible", hide_vanished=hide_vanished)
    max_df = get_interpretability_scores(embed, gene_names= gene_names, key = "OOD_max_possible", hide_vanished=hide_vanished)
    combined_df = get_interpretability_scores(embed, gene_names= gene_names, key = "OOD_combined", hide_vanished=hide_vanished)

    keep_latents = [
        k for k, v in combined_df.to_dict(orient="series").items()
        if (v.max() >= score_threshold)
        and (dim_subset is None or k in dim_subset)
    ]

    min_df = min_df[keep_latents]
    max_df = max_df[keep_latents]
    combined_df = combined_df[keep_latents]

    mins_long = min_df.reset_index().melt(
        id_vars="index",
        var_name="latent",
        value_name="min"
    ).rename(columns={"index": "gene"})

    maxs_long = max_df.reset_index().melt(
        id_vars="index",
        var_name="latent",
        value_name="max"
    ).rename(columns={"index": "gene"})

    combined_long = combined_df.reset_index().melt(
        id_vars="index",
        var_name="latent",
        value_name="combined"
    ).rename(columns={"index": "gene"})


    merged = pd.merge(mins_long, maxs_long, on=["latent", "gene"], how="outer")
    merged_final = pd.merge(merged, combined_long, on=["latent", "gene"], how="outer")
    latents_sorted = min_df.columns.tolist()

    merged_final = merged_final.reset_index(drop=True)
    merged_final["row_id"] = merged_final.index

    nrow = int(np.ceil(merged_final["latent"].nunique() / ncols))

    fig = px.scatter(
        merged_final,
        x="min",
        y="max",
        facet_col="latent",
        facet_col_wrap=ncols,
        facet_row_spacing = 0.04,
        size = "combined",
        custom_data=["row_id"],
        category_orders={"latent": latents_sorted},
        hover_name= "gene",
        hover_data={
            "min":":.3f",
            "max":":.3f",
            "combined":":.3f",
            "latent":False
    }
    )
    fig.update_yaxes(showticklabels=True, title="")
    fig.for_each_annotation(
    lambda a: a.update(
        text=a.text.split("=")[-1])
        )

    fig.update_layout(height=400 * 5,
    annotations=[
        a for a in fig.layout.annotations  # keep the facet labels
        ] + [
            dict(
                text="LFC",
                x=-0.04,            # nudge left of the plot area
                xref="paper",
                y=0.5,
                yref="paper",
                showarrow=False,
                textangle=-90,
                font=dict(size=14),
                )
            ],
        )
    fig.update_xaxes(showticklabels=True, title="Relative LFC")

    
    
    return fig, merged_final

def get_interpretability_scores(
    embed,
    gene_names,
    key: str = "OOD_combined",
    directional: bool = True,
    order_col: str = "order",
    title_col: str = "title",
    hide_vanished: bool = True,
) -> pd.DataFrame:
    
    """
    Extract interpretability scores as a DataFrame.

    This function reads interpretability scores from `embed.varm`, labels rows by
    genes, and labels columns by latent dimension titles. It can return directional
    scores, with separate positive and negative columns for each latent dimension,
    or non-directional scores with one column per latent dimension.

    Parameters
    ----------
    embed
        AnnData object containing interpretability scores in `.varm` and latent
        dimension metadata in `.var`.

    gene_names
        Gene names used as columns of the score arrays in `embed.varm`. These become
        the row index of the returned DataFrame.

    key
        Base key for the score arrays in `embed.varm`. If `directional=True`, the
        function expects `{key}_positive` and `{key}_negative`. If
        `directional=False`, it expects `key`.

    directional
        Whether to return separate positive and negative directional scores for each
        latent dimension.

    order_col
        Column in `embed.var` used to order latent dimensions.

    title_col
        Column in `embed.var` used to name latent dimensions.

    hide_vanished
        Whether to exclude vanished dimensions or vanished directions.

    Returns
    -------
    pandas.DataFrame
        DataFrame with genes as rows and latent dimensions or latent directions as
        columns.

    Notes
    -----
    When `directional=True`, the function expects `embed.var` to contain
    `"vanished_positive_direction"` and `"vanished_negative_direction"` if
    `hide_vanished=True`.

    When `directional=False`, the function expects `embed.var` to contain
    `"vanished"` if `hide_vanished=True`.

    Examples
    --------
    >>> scores = get_interpretability_scores(embed, gene_names)

    >>> scores = get_interpretability_scores(
    ...     embed,
    ...     gene_names,
    ...     key="OOD_max_possible",
    ... )

    >>> scores = get_interpretability_scores(
    ...     embed,
    ...     gene_names,
    ...     directional=False,
    ...     hide_vanished=False,
    ... )
    """

    if directional:
        effect_data = np.concatenate([embed.varm[key + "_positive"], embed.varm[key + "_negative"]])
        var_info = (
            pd.concat([embed.var.assign(direction="+"), embed.var.assign(direction="-")])
            .assign(title=lambda df: df[title_col] + df["direction"])
            .assign(keep=True)
            .reset_index(drop=True)
        )
        var_info["keep"] = (
            ~np.where(
                var_info["direction"] == "+",
                var_info["vanished_positive_direction"],
                var_info["vanished_negative_direction"],
            )
            if hide_vanished
            else True
        )
    else:
        effect_data = embed.varm[key]
        var_info = embed.var.assign(title=lambda df: df[title_col]).assign(direction="")
        var_info["keep"] = ~var_info["vanished"] if hide_vanished else True


    return (
        pd.DataFrame(
            effect_data,
            columns=gene_names,
            index=var_info["title"],
        )
        .loc[var_info.query("keep == True").sort_values([order_col, "direction"])["title"]]
        .T
    )


def plot_interpretability_scores(
        embed,
        gene_names,
        ncols: int = 5,
        n_top_genes: int = 10,
        score_threshold: float = 0.1,
        dim_subset: Sequence[str] | None = None,
        show: bool = True,
        hide_vanished = True
    ):

    """
    Plot top interpretability scores as horizontal bar plots.

    This function computes interpretability scores for latent dimensions, filters
    dimensions by a minimum score threshold, and plots the top-scoring genes for
    each selected dimension as horizontal bar plots.

    Parameters
    ----------
    embed
        AnnData object containing latent dimensions and interpretability scores.

    gene_names
        Gene annotation or list of gene names used by `get_interpretability_scores`
        to label the computed scores.

    ncols
        Number of columns in the subplot grid.

    n_top_genes
        Number of top genes to display per latent dimension.

    score_threshold
        Minimum maximum interpretability score required for a latent dimension to be
        plotted.

    dim_subset
        Optional list of latent dimension titles to plot. If None, all dimensions
        passing `score_threshold` are plotted.

    show
        Whether to display the plot with `plt.show()`. If False, the figure is
        returned.

    hide_vanished
        Whether to exclude vanished latent dimensions from the score computation.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object if `show=False`; otherwise None.

    Notes
    -----
    The function uses `get_interpretability_scores` with its default score key,
    `"OOD_combined"`.

    Examples
    --------
    >>> plot_interpretability_scores(embed, gene_names)

    >>> fig = plot_interpretability_scores(
    ...     embed,
    ...     gene_names,
    ...     show=False,
    ... )

    >>> plot_interpretability_scores(
    ...     embed,
    ...     gene_names,
    ...     n_top_genes=20,
    ...     score_threshold=0.2,
    ... )
    """

    plot_df = get_interpretability_scores(embed=embed, gene_names = gene_names, hide_vanished=hide_vanished)
    # create a list of lists where each inner list contains the latent ID and a list of its ranked genes
    # a latent is present only if its maximum score is higher than the threshold (to avoid considering meaningless latents)

    plot_info = [
        (k, v)
        for k, v in plot_df.to_dict(orient="series").items()
        if (v.max() >= score_threshold) and (dim_subset is None or k in dim_subset)
    ]

    # define the needed number of rows to fit all plots
    n_row = int(np.ceil(len(plot_info) / ncols))
    # create subplot grid ( the height of each plot depends on the number of genes displayed)
    fig, axes = plt.subplots(n_row, ncols, figsize=(3 * ncols, int(1 + 0.2 * n_top_genes) * n_row))

    for ax, info in zip(axes.flatten(), plot_info, strict=False):
        top_indices = info[1].sort_values(ascending=False)[:n_top_genes]
        if len(top_indices) > 0:
            ax.barh(top_indices.index, top_indices.values, color="skyblue")
            ax.set_xlabel("Gene Score")
            ax.set_title(info[0])
            ax.invert_yaxis()
        ax.grid(False)

    for ax in axes.flatten()[len(plot_info) :]:
        fig.delaxes(ax)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig


def plot_dr_xy_centroids(
    adata,
    DR,
    ncols=3,
    size=0.01,
    use_pheno_prefix=False,
    x_col="CenterX_global_px",
    y_col="CenterY_global_px",
    sample_col="did",
):
    """
    Plot the spatial distribution of one latent dimension across samples.

    This function extracts one latent dimension from `adata.X` and plots its values
    over spatial x-y coordinates for each sample. Each sample is shown in a separate
    subplot, using a shared color scale across all samples.

    Parameters
    ----------
    adata
        AnnData object containing latent dimension values in `.X` and spatial
        coordinate information in `.obs`.

    DR
        Zero-based index of the latent dimension to plot.

    ncols
        Number of columns to use in the subplot grid.

    size
        Scatter point size.

    use_pheno_prefix
        Whether to prepend `"pheno:"` to `x_col`, `y_col`, and `sample_col`.

    x_col
        Column in `adata.obs` containing x coordinates.

    y_col
        Column in `adata.obs` containing y coordinates.

    sample_col
        Column in `adata.obs` identifying samples.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure containing one spatial scatter plot for each sample.

    Notes
    -----
    The selected latent dimension is added to `adata.obs` as `"DR{DR}"`, where `DR`
    is the zero-based index passed to the function.

    All subplots use the same color scale, based on the minimum and maximum value of
    the selected latent dimension across the full dataset.

    Examples
    --------
    >>> fig = plot_dr_xy_centroids(adata, DR=0)

    >>> fig = plot_dr_xy_centroids(
    ...     adata,
    ...     DR=5,
    ...     ncols=4,
    ...     size=0.05,
    ... )

    >>> fig = plot_dr_xy_centroids(
    ...     adata,
    ...     DR=0,
    ...     use_pheno_prefix=True,
    ... )
    """

    dr = int(DR)
    dr_name = f"DR{dr}"

    prefix = "pheno:" if use_pheno_prefix else ""

    x_col = f"{prefix}{x_col}"
    y_col = f"{prefix}{y_col}"
    sample_col = f"{prefix}{sample_col}"

    required_cols = [x_col, y_col, sample_col]
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]

    if missing_cols:
        raise KeyError(
            f"Missing required columns in adata.obs: {missing_cols}. "
            f"Available columns include: {list(adata.obs.columns[:20])}"
        )

    # Add selected DR component to adata.obs
    adata.obs[dr_name] = adata.X[:, dr]

    # Store spatial coordinates
    adata.obsm["spatial"] = adata.obs[[x_col, y_col]].to_numpy()

    vmin = adata.obs[dr_name].min()
    vmax = adata.obs[dr_name].max()

    samples = adata.obs[sample_col].unique()
    nrows = int(np.ceil(len(samples) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    last_sc = None

    for ax, sample in zip(axes.ravel(), samples):
        mask = adata.obs[sample_col] == sample

        coords = adata.obsm["spatial"][mask, :]
        values = adata.obs.loc[mask, dr_name].to_numpy()

        last_sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            s=size,
            cmap="bwr",
            marker="o",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
            linewidths=0,
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"Sample: {sample}", ha="center")
        

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("")

    for ax in axes.ravel()[len(samples):]:
        ax.axis("off")

    return fig


def plot_sample_all_latents_xy_centroids(
    adata,
    sample,
    use_pheno_prefix=False,
    sample_col="batch",
    x_col="CenterX_global_px",
    y_col="CenterY_global_px",
    class_col="lv2",
    ncols=4,
    size=0.01,
    cmap_latent="bwr",
    discrete_cmap="rainbow",
):
    """
    Plot all latent dimensions for one sample in spatial coordinates.

    This function selects one sample from `adata`, then plots every latent dimension
    from `adata.X` over the sample's spatial x-y coordinates. It also adds one extra
    panel showing a categorical annotation, such as a cell type or class label, over
    the same coordinates.

    Parameters
    ----------
    adata
        AnnData object containing latent dimension values in `.X` and spatial/sample
        metadata in `.obs`.

    sample
        Sample identifier to plot. The value is matched against `sample_col`.

    use_pheno_prefix
        Whether to prepend `"pheno:"` to `sample_col`, `x_col`, `y_col`, and
        `class_col`.

    sample_col
        Column in `adata.obs` identifying samples.

    x_col
        Column in `adata.obs` containing x coordinates.

    y_col
        Column in `adata.obs` containing y coordinates.

    class_col
        Categorical column in `adata.obs` to display in the extra annotation panel.

    ncols
        Number of columns to use in the subplot grid.

    size
        Scatter point size.

    cmap_latent
        Matplotlib colormap used for latent dimension values.

    discrete_cmap
        Matplotlib colormap used for the categorical annotation panel.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure containing one spatial plot per latent dimension plus one
        categorical annotation panel.

    Notes
    -----
    Latent panels are scaled using the global minimum and maximum of each latent
    dimension across all observations in `adata`, not only the selected sample.

    If `adata.X` is sparse, it is converted to a dense array before plotting.

    The function requires `sample_col`, `x_col`, and `y_col` to exist in
    `adata.obs`. The `class_col` is also required for the categorical annotation
    panel.

    Examples
    --------
    >>> fig = plot_sample_all_latents_xy_centroids(adata, sample="sample_1")

    >>> fig = plot_sample_all_latents_xy_centroids(
    ...     adata,
    ...     sample="sample_1",
    ...     sample_col="did",
    ...     class_col="lv2",
    ...     ncols=5,
    ... )

    >>> fig = plot_sample_all_latents_xy_centroids(
    ...     adata,
    ...     sample="sample_1",
    ...     use_pheno_prefix=True,
    ... )
    """

    prefix = "pheno:" if use_pheno_prefix else ""
    x_col = f"{prefix}{x_col}"
    y_col = f"{prefix}{y_col}"
    sample_col = f"{prefix}{sample_col}"
    class_col = f"{prefix}{class_col}"

    required_cols = [x_col, y_col, sample_col]
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]

    if missing_cols:
        raise KeyError(
            f"Missing required columns in adata.obs: {missing_cols}. "
            f"Available columns include: {list(adata.obs.columns[:20])}"
        )

    mask = adata.obs[sample_col] == sample

    coords = adata.obs.loc[mask, [x_col, y_col]].to_numpy()

    X_sample = adata.X[mask, :]
    if hasattr(X_sample, "toarray"):
        X_sample = X_sample.toarray()

    X_all = adata.X
    if hasattr(X_all, "toarray"):
        X_all = X_all.toarray()

    latent_count = X_all.shape[1]

    n_panels = latent_count + 1
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    axes_flat = axes.ravel()

    latent_vmins = np.nanmin(X_all, axis=0)
    latent_vmaxs = np.nanmax(X_all, axis=0)

    for dr in range(latent_count):
        ax = axes_flat[dr]
        values = X_sample[:, dr]

        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            s=size,
            cmap=cmap_latent,
            marker="o",
            vmin=latent_vmins[dr],
            vmax=latent_vmaxs[dr],
            rasterized=True,
            linewidths=0,
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"DR {dr + 1}", ha="center")

        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("")

    # Extra panel: categorical lv2 annotation with legend
    ax = axes_flat[latent_count]

    class_values = adata.obs.loc[mask, class_col].astype("category")
    categories = class_values.cat.categories
    codes = class_values.cat.codes.to_numpy()

    base_cmap = plt.get_cmap(discrete_cmap)
    colors = base_cmap(np.linspace(0, 1, len(categories)))
    listed_cmap = mcolors.ListedColormap(colors)

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=codes,
        s=size,
        cmap=listed_cmap,
        marker="o",
        rasterized=True,
        linewidths=0,
        vmin=-0.5,
        vmax=len(categories) - 0.5,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(class_col, ha="center")

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[i],
            markeredgecolor="none",
            markersize=6,
            label=str(cat),
        )
        for i, cat in enumerate(categories)
    ]

    ax.legend(
        handles=legend_handles,
        title=class_col,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=7,
        title_fontsize=8,
    )

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.suptitle(f"Sample: {sample}", fontsize=12)

    return fig
