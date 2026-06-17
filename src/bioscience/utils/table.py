import anndata as ad
import numpy as np
import pandas as pd
import scanpy 
from scipy import sparse, stats
from bioscience.utils import plotting
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from plottable import Table
from plottable import ColumnDefinition

def morans_i_per_sample(
    adata,
    sample_col="did",
    sample_value=None,
    spatial_cols=("CenterX_global_px", "CenterY_global_px"),
    n_neighbors=15,
    use_pheno_prefix=False,
):
    """
    Compute Moran's I spatial autocorrelation for latent dimensions per sample.

    This function computes Moran's I for each latent dimension in `adata.X`,
    separately for one or more samples. For each sample, a spatial nearest-neighbor
    graph is constructed from x-y coordinate columns in `adata.obs`, and Moran's I
    is computed using the resulting spatial connectivities.

    The function returns both the per-sample Moran's I values and a summary table
    reporting, for each latent dimension, the samples where Moran's I reaches its
    minimum and maximum values.

    Parameters
    ----------
    adata
        AnnData object containing latent dimension values in `.X` and sample/spatial
        metadata in `.obs`.

    sample_col
        Column in `adata.obs` identifying the sample or dataset each observation
        belongs to. If `use_pheno_prefix=True`, `"pheno:"` is prepended to this
        column name.

    sample_value
        Optional sample identifier. If provided, Moran's I is computed only for this
        sample. If None, Moran's I is computed for all samples in `sample_col`.

    spatial_cols
        Pair of columns in `adata.obs` containing the spatial x-y coordinates used
        to build the neighbor graph. If `use_pheno_prefix=True`, `"pheno:"` is
        prepended to each coordinate column name.

    n_neighbors
        Number of spatial nearest neighbors to use when constructing the spatial
        connectivity graph.

    use_pheno_prefix
        Whether to prepend `"pheno:"` to `sample_col` and all columns in
        `spatial_cols` before looking them up in `adata.obs`.

    Returns
    -------
    pandas.DataFrame
        Long-format table containing Moran's I values for each sample and latent
        dimension. The table includes the sample identifier, zero-based latent index,
        latent name, Moran's I value, and number of cells in the sample.

    pandas.DataFrame
        Summary table indexed by latent dimension. For each latent dimension, it
        reports the maximum Moran's I, the sample where that maximum occurs, the
        minimum Moran's I, and the sample where that minimum occurs.

    Notes
    -----
    The function expects `adata.obs` to contain the column specified by `sample_col`
    and the coordinate columns specified by `spatial_cols`, after optional
    `"pheno:"` prefixing. If any required columns are missing, a `KeyError` is
    raised.

    Latent dimensions are named as `"DR1"`, `"DR2"`, ..., based on the number of
    columns in `adata.X`. The zero-based latent index is also stored in the returned
    per-sample table.

    If `adata.X` is sparse, it is converted to a dense array before computing
    Moran's I.

    Examples
    --------
    >>> moran_by_sample, summary = morans_i_per_sample(adata)

    >>> # Compute Moran's I for a single sample
    >>> moran_by_sample, summary = morans_i_per_sample(
    ...     adata,
    ...     sample_value="sample_1",
    ... )

    >>> # Use custom spatial coordinate columns and neighbor count
    >>> moran_by_sample, summary = morans_i_per_sample(
    ...     adata,
    ...     spatial_cols=("x_coord", "y_coord"),
    ...     n_neighbors=20,
    ... )

    >>> # Use columns stored with a "pheno:" prefix
    >>> moran_by_sample, summary = morans_i_per_sample(
    ...     adata,
    ...     use_pheno_prefix=True,
    ... )
    """

    prefix = "pheno:" if use_pheno_prefix else ""

    sample_col = f"{prefix}{sample_col}"
    spatial_cols = tuple(f"{prefix}{col}" for col in spatial_cols)

    required_cols = [sample_col, *spatial_cols]
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in adata.obs: {missing_cols}")

    if sample_value is None:
        sample_ids = adata.obs[sample_col].unique()
    else:
        sample_ids = [sample_value]

    latent_names = [f"DR{i}" for i in range(1, adata.X.shape[1] + 1)]
    records = []

    for sample_id in sample_ids:
        mask = adata.obs[sample_col] == sample_id

        adata_n = ad.AnnData(obs=adata.obs.loc[mask].copy())

        X = adata.X[mask, :]
        if hasattr(X, "toarray"):
            X = X.toarray()
        else:
            X = np.asarray(X)

        adata_n.obsm["DR"] = X
        adata_n.obsm["spatial"] = adata.obs.loc[mask, list(spatial_cols)].to_numpy()

        scanpy.pp.neighbors(
            adata=adata_n,
            n_neighbors=n_neighbors,
            use_rep="spatial",
            key_added="spatial",
        )

        moran_vals = scanpy.metrics.morans_i(
            adata_n,
            obsm="DR",
            use_graph="spatial_connectivities",
        )

        moran_vals = np.asarray(moran_vals).ravel()

        for latent_idx, val in enumerate(moran_vals):
            records.append(
                {
                    "sample": sample_id,
                    "latent_idx": latent_idx,
                    "latent": latent_names[latent_idx],
                    "moran_i": float(val),
                    "n_cells": adata_n.n_obs,
                }
            )

    moran_by_sample = pd.DataFrame(records)

    summary = (
        moran_by_sample.groupby("latent", as_index=True)
        .apply(
            lambda g: pd.Series(
                {
                    "max_moran_i": g.loc[g["moran_i"].idxmax(), "moran_i"],
                    "sample_max": g.loc[g["moran_i"].idxmax(), "sample"],
                    "min_moran_i": g.loc[g["moran_i"].idxmin(), "moran_i"],
                    "sample_min": g.loc[g["moran_i"].idxmin(), "sample"],
                }
            )
        )
    )

    return moran_by_sample, summary


def max_adjusted_r2_per_latent(
    adata,
    label_cols=("did", "lv2"),
    latent_key=None,
    use_pheno_prefix=False,
):
    """
    For each latent, find the category with the maximum one-vs-rest adjusted R^2.

    This helper evaluates a continuous latent against each category of the
    requested categorical columns separately. The association is measured as adjusted R-squared from the Pearson
correlation between the latent values and the category indicator. It is useful for asking which
    sample id or lv2 category is most strongly associated with each latent.

    Parameters
    ----------
    adata
        AnnData object containing latent values in `.X` or `.obsm[latent_key]`
        and categorical labels in `.obs`.

    label_cols
        Iterable of categorical columns to evaluate, by default both `did` and
        `lv2`.

    latent_key
        Optional key in `adata.obsm` for the latent matrix. If None, uses
        `adata.X`.

    use_pheno_prefix
        Whether to prepend `"pheno:"` to the requested label column names before
        looking them up in `adata.obs`.

    Returns
    -------
    pandas.DataFrame
        One row per latent and per requested label column, with the best category
        (highest adjusted R^2), the adjusted R^2 value, and the group size.
    
    Notes
    -----
    Categories containing all or none of the observations are skipped. Latents or
    category indicators with zero variance are also skipped.

    Latent dimensions are named `"DR1"`, `"DR2"`, and so on. Sparse latent matrices
    are converted to dense arrays before calculation.

    Examples
    --------
    >>> r2_summary = max_adjusted_r2_per_latent(adata)

    >>> r2_summary = max_adjusted_r2_per_latent(
    ...     adata,
    ...     label_cols=("did", "lv2", "condition"),
    ... )

    >>> r2_summary = max_adjusted_r2_per_latent(
    ...     adata,
    ...     latent_key="X_dr",
    ...     use_pheno_prefix=True,
    ... )
    """

    prefix = "pheno:" if use_pheno_prefix else ""
    label_cols = [f"{prefix}{col}" for col in label_cols]

    X = adata.obsm[latent_key] if latent_key is not None else adata.X
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    missing_cols = [col for col in label_cols if col not in adata.obs.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in adata.obs: {missing_cols}")

    latent_names = [f"DR{i}" for i in range(1, X.shape[1] + 1)]
    records = []

    for label_col in label_cols:
        labels = adata.obs[label_col]
        valid_mask = labels.notna().to_numpy()
        labels = labels.loc[valid_mask].astype(str).to_numpy()
        X_label = X[valid_mask, :]
        n_obs_label = X_label.shape[0]
        categories = pd.unique(labels)

        for latent_idx, latent_name in enumerate(latent_names):
            latent = X_label[:, latent_idx]

            best_category = None
            best_adj_r2 = np.nan
            best_n_cells = np.nan

            for category in categories:
                indicator = (labels == category).astype(float)
                n_category = int(indicator.sum())

                if n_category == 0 or n_category == n_obs_label:
                    continue
                if np.nanstd(latent) == 0 or np.nanstd(indicator) == 0:
                    continue

                corr = stats.pearsonr(latent, indicator)[0]
                r2 = float(corr**2)
                adj_r2 = (
                    1 - (1 - r2) * (n_obs_label - 1) / (n_obs_label - 2)
                    if n_obs_label > 2
                    else np.nan
                )

                if np.isnan(best_adj_r2) or adj_r2 > best_adj_r2:
                    best_category = category
                    best_adj_r2 = float(adj_r2)
                    best_n_cells = n_category

            records.append(
                {
                    "label_col": label_col,
                    "latent_idx": latent_idx,
                    "latent": latent_name,
                    "best_category": best_category,
                    "adj_r2": best_adj_r2,
                    "n_cells_in_best_category": best_n_cells,
                }
            )

    result = pd.DataFrame(records)
    return result.sort_values(["label_col", "adj_r2"], ascending=[True, False])


def plot_interpretability_metric_table(
    embed,
    gene_names,
    figsize=(15, 35),
    hide_vanished=True,
):
    
    """
    Plot a table of normalized interpretability metrics for latent dimensions.

    This function computes interpretability scores for each latent dimension using
    the maximum possible, minimum possible, and combined out-of-distribution score
    definitions. For each score type, the maximum score per latent dimension is
    normalized by the global maximum score and displayed in a formatted table.

    Parameters
    ----------
    embed
        AnnData object containing latent dimensions and interpretability information.

    gene_names
        Gene names or gene annotation passed to
        `plotting.get_interpretability_scores`.

    figsize
        Size of the Matplotlib figure used to draw the table.

    hide_vanished
        Whether to exclude vanished latent dimensions when computing
        interpretability scores.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure containing the formatted interpretability metric table.

    Notes
    -----
    The function uses the score keys `"OOD_max_possible"`, `"OOD_min_possible"`, and
    `"OOD_combined"`.

    Scores are normalized separately for each score key by dividing each latent's
    maximum score by the largest maximum score across all latents for that key.

    Examples
    --------
    >>> fig = plot_interpretability_metric_table(embed, gene_names)

    >>> fig = plot_interpretability_metric_table(
    ...     embed,
    ...     gene_names,
    ...     figsize=(12, 25),
    ...     hide_vanished=False,
    ... )
    """

    score_keys = [
        "OOD_max_possible",
        "OOD_min_possible",
        "OOD_combined",
    ]

    def summarize_scores(key):
        scores = plotting.get_interpretability_scores(
            embed=embed,
            gene_names=gene_names,
            key=key,
            hide_vanished=hide_vanished,
        )

        df = pd.DataFrame({
            #f"gene_{key}": scores.idxmax(),
            f"{key}_normalized": scores.max() / scores.max().max(),
        })

        return df

    metric_df = (
        pd.concat([summarize_scores(key) for key in score_keys], axis=1)
        .reset_index()
        .rename(columns={"index": "title"})
        .round(3)
    )

    display_columns = [
        "title",
        #"gene_OOD_max_possible",
        "OOD_max_possible_normalized",
        #"gene_OOD_min_possible",
        "OOD_min_possible_normalized",
        #"gene_OOD_combined",
        "OOD_combined_normalized",
    ]

    metric_df = metric_df[display_columns]


    circle_textprops = {
        "ha": "center",
        "bbox": {"boxstyle": "circle", "pad": 0.3},
    }

    norm = mcolors.Normalize(vmin=0, vmax=1.3)
    cmap = cm.Reds

    def color(x):
        return cmap(norm(x))

    fig, ax = plt.subplots(figsize=figsize)

    Table(
        df=metric_df,
        ax=ax,
        index_col="title",
        column_definitions=[
            ColumnDefinition(
                name="title",
                title="Latent",
                textprops={"ha": "left", "weight": "bold", "fontsize": 11},
            ),
            #ColumnDefinition(
            #    name="gene_OOD_max_possible",
            #    title="Max gene",
            #    width=0.7,
            #    textprops={"fontsize": 11},
            #),
            ColumnDefinition(
                name="OOD_max_possible_normalized",
                title="Max score",
                width=0.7,
                textprops=circle_textprops,
                cmap=color,
                formatter="{:.3f}"
            ),
            #ColumnDefinition(
            #    name="gene_OOD_min_possible",
            #    title="Min gene",
            #    width=0.7,
            #    textprops={"fontsize": 11},
            #),
            ColumnDefinition(
                name="OOD_min_possible_normalized",
                title="Min score",
                width=0.7,
                textprops=circle_textprops,
                cmap=color,
                formatter="{:.3f}"
            ),
         #   ColumnDefinition(
         #       name="gene_OOD_combined",
         #       title="Combined gene",
         #       width=0.7,
         #       textprops={"fontsize": 11},
         #   ),
            ColumnDefinition(
                name="OOD_combined_normalized",
                title="Combined score",
                width=0.7,
                textprops=circle_textprops,
                cmap=color,
                formatter="{:.3f}"
            ),
        ],
        row_dividers=True,
        row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
    )


    return fig

def plot_latent_correlation_table(embed, 
                                  latent_names=None, 
                                  moran_summary: pd.DataFrame | None = None,
                                  r2_summary: pd.DataFrame | None = None,
                                  use_pheno_prefix: bool = False,
                                  figsize: tuple[float, float] = (15, 20)):
    
    """
    Plot a summary table of latent correlations and optional spatial/annotation metrics.

    This function computes pairwise correlations between latent dimensions in
    `embed.X`. For each latent dimension, it identifies the other latent dimension
    with the strongest absolute correlation and displays the result in a formatted
    table. Optional adjusted R-squared summaries and Moran's I summaries can be
    merged into the same table.

    Parameters
    ----------
    embed
        AnnData object containing latent dimension values in `.X`.

    latent_names
        Optional names for the latent dimensions. If None, latent dimensions are
        named `"DR1"`, `"DR2"`, and so on.

    moran_summary
        Optional summary DataFrame returned by `morans_i_per_sample`. If provided,
        Moran's I maximum and minimum values, together with the corresponding sample
        identifiers, are added to the table.

    r2_summary
        Optional summary DataFrame returned by `max_adjusted_r2_per_latent`. If
        provided, the best associated categories and adjusted R-squared values are
        added to the table.

    use_pheno_prefix
        Whether `"pheno:"` should be removed from label column names in `r2_summary`
        before reshaping the table.

    figsize
        Size of the Matplotlib figure used to draw the table.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure containing the latent correlation summary table.

    Notes
    -----
    Self-correlations are ignored when identifying the strongest correlated latent
    dimension.

    The displayed correlation value is the absolute correlation between each latent
    and its strongest non-self match.

    The function expects `embed.X` to be a dense numeric matrix.

    Examples
    --------
    >>> fig = plot_latent_correlation_table(embed)

    >>> moran_by_sample, moran_summary = morans_i_per_sample(adata)
    >>> r2_summary = max_adjusted_r2_per_latent(adata)
    >>> fig = plot_latent_correlation_table(
    ...     embed,
    ...     moran_summary=moran_summary,
    ...     r2_summary=r2_summary,
    ... )
    """

    matrix = embed.X
    corr = np.round(np.corrcoef(matrix, rowvar=False), 3)

    n_latents = corr.shape[0]
    if latent_names is None:
        latent_names = [f"DR{i}" for i in range(1, n_latents + 1)]

    corr_df = pd.DataFrame(corr, index=latent_names, columns=latent_names)

    # Remove self-correlations before finding the strongest match.
    corr_no_diag = corr_df.mask(np.eye(n_latents, dtype=bool))

    strongest_latent = corr_no_diag.abs().idxmax(axis=1)
    strongest_corr = [
        abs(corr_no_diag.loc[latent, match])
        for latent, match in strongest_latent.items()
    ]

    result = pd.DataFrame({
        "latent": strongest_latent.index,
        "most_correlated_with": strongest_latent.values,
        "correlation": strongest_corr,
    })

    if r2_summary is not None:
        prefix = "pheno:" if use_pheno_prefix else ""
        r2_summary = r2_summary.copy()
        r2_summary["label_col"] = r2_summary["label_col"].str.removeprefix(prefix)

        r2_best = (
            r2_summary.pivot(index="latent", columns="label_col", values="best_category")
            .reset_index()
            .rename(columns={"did": "did_best_category", "lv2": "lv2_best_category"})
        )
        r2_corr = (
            r2_summary.pivot(index="latent", columns="label_col", values="adj_r2")
            .reset_index()
            .rename(columns={"did": "did_adj_r2", "lv2": "lv2_adj_r2"})
        )

        result = result.merge(r2_best, on="latent", how="left")
        result = result.merge(r2_corr, on="latent", how="left")


    if moran_summary is not None:
            result = result.merge(moran_summary, on="latent", how="left")

    def _format_sample_id(value):
        if pd.isna(value):
            return value
        if isinstance(value, (int, np.integer)):
            return str(value)
        if isinstance(value, (float, np.floating)) and value.is_integer():
            return str(int(value))
        return str(value)



    for column in ("sample_max", "sample_min"):
        if column in result.columns:
            result[column] = result[column].map(_format_sample_id)

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = matplotlib.cm.Reds

    def correlation_color(x):
        return cmap(norm(x))

    norm2 = mcolors.Normalize(vmin=0, vmax=1.3)
    cmap2 = cm.Blues

    def color2(x):
        return cmap2(norm2(x))

    fig, ax = plt.subplots(figsize=figsize)

    # enforce explicit column display order
    desired_columns = [
        "latent",
        "most_correlated_with",
        "correlation",
        "did_best_category",
        "did_adj_r2",
        "lv2_best_category",
        "lv2_adj_r2",
        "max_moran_i",
        "sample_max",
        "min_moran_i",
        "sample_min",
    ]

    # keep only columns that exist in `result`, preserving desired order
    existing = [c for c in desired_columns if c in result.columns]
    other = [c for c in result.columns if c not in existing]
    result = result[existing + other]

    column_definitions = [
            ColumnDefinition(
                name="latent",
                textprops={"ha": "left", "weight": "bold",},
            ),
            ColumnDefinition(
                name="most_correlated_with",
                width=0.4
            ),
            ColumnDefinition(
                name="correlation",
                width=0.7,
                textprops={
                    "ha": "center",
                    "bbox": {"boxstyle": "circle", "pad": 0.05},
                },
                cmap=correlation_color,
                formatter="{:.3f}",
            ),
        ]

    if "did_best_category" in result.columns:
            column_definitions.append(
                ColumnDefinition(
                    name="did_best_category",
                    width=0.65,
                )
            )

    if "did_adj_r2" in result.columns:
        column_definitions.append(
            ColumnDefinition(
                name="did_adj_r2",
                width=0.6,
                textprops={
                    "ha": "center",
                    "bbox": {"boxstyle": "circle", "pad": 0.05},
                },
                cmap=color2,
                formatter="{:.3f}",
            )
        )

    if "lv2_best_category" in result.columns:
        column_definitions.append(
            ColumnDefinition(
                name="lv2_best_category",
                width=0.65,
            )
        )

    if "lv2_adj_r2" in result.columns:
        column_definitions.append(
            ColumnDefinition(
                name="lv2_adj_r2",
                width=0.6,
                textprops={
                    "ha": "center",
                    "bbox": {"boxstyle": "circle", "pad": 0.05},
                },
                cmap=color2,
                formatter="{:.3f}",
            )
        )

    if "max_moran_i" in result.columns:
            column_definitions.append(
                ColumnDefinition(
                    name="max_moran_i",
                    width=0.7,
                    textprops={"ha": "center",
                              "bbox": {"boxstyle": "circle", "pad": 0.05},},
                    formatter="{:.3f}",
                    cmap = color2
                )
            )

    if "sample_max" in result.columns:
        column_definitions.append(
            ColumnDefinition(
                name="sample_max",
                width=0.5,
            )
        )

    if "min_moran_i" in result.columns:
        column_definitions.append(
            ColumnDefinition(
                name="min_moran_i",
                width=0.7,
                textprops={"ha": "center",
                          "bbox": {"boxstyle": "circle", "pad": 0.05},},
                formatter="{:.3f}",
                cmap = color2
            )
        )

    if "sample_min" in result.columns:
        column_definitions.append(
            ColumnDefinition(
                name="sample_min",
                width=0.5,
            )
        )

    Table(
            df=result,
            ax=ax,
            index_col="latent",
            column_definitions=column_definitions,
            row_dividers=True,
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
        )

    return fig
