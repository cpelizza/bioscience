from bioscience.utils import plotting, table
import anndata as ad
import os
from pyprojroot import here
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

parser = ArgumentParser()
parser.add_argument(
    "--h5ad_path",
    type=str,
    help="Name of the embedding file inside the data folder",
)
parser.add_argument(
    "--gene_path",
    type=str,
    help="Name of the gene file inside the data folder",
)
args = parser.parse_args()

h5ad_path = args.h5ad_path or input("Enter h5ad file name: ").strip()
gene_path = args.gene_path or input("Enter gene file name: ").strip()

h5ad_name = os.path.splitext(os.path.basename(h5ad_path))[0]

output_base_folder = here("output")

if not os.path.exists(output_base_folder):
    os.makedirs(output_base_folder)
    print(f"Created output folder: {output_base_folder}", flush=True)
else:
    print(f"Output folder already exists: {output_base_folder}", flush=True)

output_folder = os.path.join(output_base_folder, h5ad_name)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created sample output folder: {output_folder}", flush=True)
else:
    print(f"Sample output folder already exists: {output_folder}", flush=True)

print(f"Saving outputs to: {output_folder}", flush=True)

#load embed data
print(f"Loading h5ad file: {h5ad_path}", flush=True)
adata = ad.read_h5ad(h5ad_path, backed="r")

#load gene names
print(f"Loading gene file: {gene_path}", flush=True)
with open(gene_path, "r", encoding = "utf-8") as f:
    text = f.read()

#make it as a list
genes = [g.strip().strip("'") for g in text.split(",")]

use_prefix = input("Use prefix for column names? (y/n): ").strip().lower() == "y"

# spatial plots
latent_count = adata.X.shape[1]
latent_numbers = list(range(latent_count))

pdf_filename = os.path.join(output_folder, f"{h5ad_name}-spatial_plots.pdf")

print(f"Creating spatial plots for {latent_count} latents...", flush=True)
with PdfPages(pdf_filename) as pdf:
    for i, latent_number in enumerate(latent_numbers, start=1):
        print(f"  Spatial plot {i}/{latent_count}: DR {latent_number + 1}", flush=True)

        fig = plotting.plot_dr_xy_centroids(
            adata=adata,
            DR=latent_number,
            size=0.1,
            use_pheno_prefix=use_prefix,
        )

        fig.suptitle(f"DR {latent_number + 1}", fontsize=12)
        pdf.savefig(fig, bbox_inches="tight", dpi=300)
        plt.close(fig)

print(f"Saved: {pdf_filename}", flush=True)


# per sample plot
pdf_filename2 = os.path.join(output_folder, f"{h5ad_name}-spatial_plots_by_sample.pdf")

samples = adata.obs["batch"].unique()
sample_count = len(samples)

print(f"Creating per-sample plots for {sample_count} samples...", flush=True)
with PdfPages(pdf_filename2) as pdf:
    for i, sample in enumerate(samples, start=1):
        print(f"  Sample plot {i}/{sample_count}: {sample}", flush=True)

        fig = plotting.plot_sample_all_latents_xy_centroids(
            adata=adata,
            sample=sample,
            ncols=4,
            size=0.1,
            use_pheno_prefix=use_prefix,
        )

        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)

print(f"Saved: {pdf_filename2}", flush=True)
print("Done.", flush=True)
