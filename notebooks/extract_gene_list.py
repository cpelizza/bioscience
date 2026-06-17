import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import anndata as ad
    from pyprojroot import here
    import os

    return ad, here, os


@app.cell
def _(here):
    here()
    return


@app.cell
def _(ad):
    adata = ad.read_h5ad("/mnt/lustre/scratch/nlsas/home/res/cnag71/resh000982/bioscience/data/full_object_phenotype.h5ad")
    return (adata,)


@app.cell
def _(adata):
    genes = adata.var_names.to_list()
    return (genes,)


@app.cell
def _(genes, here, os):
    with open(os.path.join(here(), "results/L40_S123_260611111342/genes.txt"), "w") as f:
        for gene in genes:
            f.write(gene + "\n")

    print("File creato: genes.txt")
    return


if __name__ == "__main__":
    app.run()
