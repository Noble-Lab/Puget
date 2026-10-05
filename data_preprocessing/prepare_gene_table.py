from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
from tqdm.auto import tqdm


BOUNDARY_HALF_BP = 262_144
MAPPABILITY_HALF_BP = 196_608 // 2
MAP_THRESHOLD = 0.5
MAX_UNMAPPABLE_FRACTION = 0.5
CANONICAL_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX"}
GENE_ID_RE = re.compile(r'gene_id "([^"]+)"')
GENE_NAME_RE = re.compile(r'gene_name "([^"]+)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epinformer-csv", type=Path, required=True,
                        help="EPInformer 18,377-gene annotation CSV")
    parser.add_argument("--gencode", type=Path, required=True,
                        help="GENCODE v29 hg38 annotation GTF")
    parser.add_argument("--chrom-sizes", type=Path, required=True,
                        help="hg38 chromosome sizes, two-column text file")
    parser.add_argument("--mappability-bw", type=Path, required=True,
                        help="k50.Umap.MultiTrackMappability.bw for hg38")
    parser.add_argument("--biosample-csv", type=Path, required=True,
                        help="CSV with RNA-seq accession and Hi-C accession columns")
    parser.add_argument("--rna-dir", type=Path, required=True,
                        help="Directory containing <RNA-seq accession>.tsv quantifications")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Filtered gene table with one expression column per Hi-C accession")
    return parser.parse_args()


def read_gencode_genes(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            gene_id = GENE_ID_RE.search(fields[8])
            gene_name = GENE_NAME_RE.search(fields[8])
            if gene_id is None or gene_name is None:
                raise ValueError(f"{path}:{line_number}: gene lacks gene_id or gene_name")
            versioned_id = gene_id.group(1)
            rows.append((versioned_id.split(".")[0], versioned_id,
                         gene_name.group(1), fields[0], fields[6]))
    return pd.DataFrame(rows, columns=[
        "gene_id", "gene_id_v", "gc_name", "gc_chrom", "gc_strand"
    ])


def read_chrom_sizes(path: Path) -> dict[str, int]:
    sizes = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, size = line.split()[:2]
            sizes[chrom] = int(size)
    return sizes


def main() -> None:
    args = parse_args()
    for path in (args.epinformer_csv, args.gencode, args.chrom_sizes,
                 args.mappability_bw, args.biosample_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.rna_dir.is_dir():
        raise NotADirectoryError(args.rna_dir)

    epi = pd.read_csv(args.epinformer_csv)
    required_epi = {"gene_id", "gene_name", "chrom", "strand", "TSS"}
    if missing := required_epi.difference(epi.columns):
        raise ValueError(f"{args.epinformer_csv}: missing columns {sorted(missing)}")
    epi["gene_id"] = epi["gene_id"].astype(str)
    epi["chrom"] = "chr" + epi["chrom"].astype(str)
    print(f"EPInformer genes: {len(epi)}", flush=True)

    gencode = read_gencode_genes(args.gencode)
    merged = epi.merge(gencode, on="gene_id", how="inner")
    agree = (
        (merged["gc_name"] == merged["gene_name"])
        & (merged["gc_chrom"] == merged["chrom"])
        & (merged["gc_strand"] == merged["strand"])
    )
    genes = merged.loc[agree].drop_duplicates("gene_id").reset_index(drop=True)
    print(f"GENCODE v29 genes: {len(gencode)}; strict matches: {len(genes)}", flush=True)

    before = len(genes)
    genes = genes[genes["chrom"].isin(CANONICAL_CHROMS)].reset_index(drop=True)
    print(f"Canonical chromosomes: {len(genes)} (dropped {before - len(genes)})", flush=True)

    chrom_lengths = genes["chrom"].map(read_chrom_sizes(args.chrom_sizes))
    if chrom_lengths.isna().any():
        raise ValueError("Chromosome sizes are missing for retained genes")
    tss = genes["TSS"].to_numpy()
    near_boundary = ((tss <= BOUNDARY_HALF_BP)
                     | ((chrom_lengths.to_numpy() - tss) <= BOUNDARY_HALF_BP))
    before = len(genes)
    genes = genes.loc[~near_boundary].reset_index(drop=True)
    print(f"Chromosome boundary: {len(genes)} (dropped {before - len(genes)})", flush=True)

    bw = pyBigWig.open(str(args.mappability_bw))
    if bw is None:
        raise OSError(f"Could not open {args.mappability_bw}")
    try:
        pct_unmapped = np.empty(len(genes))
        for i, row in enumerate(tqdm(genes.itertuples(index=False),
                                     total=len(genes), desc="Mappability",
                                     disable=not sys.stderr.isatty())):
            start = int(row.TSS - MAPPABILITY_HALF_BP)
            end = int(row.TSS + MAPPABILITY_HALF_BP)
            values = np.nan_to_num(bw.values(row.chrom, start, end, numpy=True), nan=0.0)
            pct_unmapped[i] = np.mean(values < MAP_THRESHOLD)
    finally:
        bw.close()
    genes["pct_unmapped"] = pct_unmapped
    before = len(genes)
    genes = genes.loc[genes["pct_unmapped"] <= MAX_UNMAPPABLE_FRACTION].reset_index(drop=True)
    print(f"Mappability: {len(genes)} (dropped {before - len(genes)})", flush=True)

    bios = pd.read_csv(args.biosample_csv)
    required_bios = {"RNA-seq accession", "Hi-C accession"}
    if missing := required_bios.difference(bios.columns):
        raise ValueError(f"{args.biosample_csv}: missing columns {sorted(missing)}")
    if bios[["RNA-seq accession", "Hi-C accession"]].isna().any().any():
        raise ValueError(f"{args.biosample_csv}: missing accession")
    if bios["RNA-seq accession"].duplicated().any() or bios["Hi-C accession"].duplicated().any():
        raise ValueError(f"{args.biosample_csv}: duplicate accession")

    expr = genes[["gene_id_v"]].copy()
    hic_accessions = []
    for rna_acc, hic_acc in bios[["RNA-seq accession", "Hi-C accession"]].itertuples(
        index=False, name=None
    ):
        rna_acc, hic_acc = str(rna_acc), str(hic_acc)
        rna_path = args.rna_dir / f"{rna_acc}.tsv"
        if not rna_path.is_file():
            raise FileNotFoundError(rna_path)
        rna = pd.read_csv(rna_path, sep="\t", usecols=["gene_id", "TPM"])
        rna = rna.rename(columns={"gene_id": "gene_id_v"})
        if rna["gene_id_v"].duplicated().any() or rna["TPM"].isna().any() or (rna["TPM"] < 0).any():
            raise ValueError(f"{rna_path}: duplicate gene IDs or invalid TPM")
        rna[hic_acc] = np.log2(rna["TPM"] + 1.0)
        expr = expr.merge(rna[["gene_id_v", hic_acc]], on="gene_id_v", how="inner")
        hic_accessions.append(hic_acc)

    before = len(genes)
    genes = genes.merge(expr, on="gene_id_v", how="inner").reset_index(drop=True)
    if genes[hic_accessions].isna().any().any():
        raise ValueError("RNA expression contains NaN")
    if not genes["gene_id"].is_unique:
        raise ValueError("Gene ID occurs more than once")
    print(f"RNA intersection: {len(genes)} (dropped {before - len(genes)})", flush=True)

    final_cols = ["gene_id", "gene_id_v", "gene_name", "chrom", "strand",
                  "TSS", "pct_unmapped"] + hic_accessions
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    genes[final_cols].to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}: {len(genes)} genes, {len(hic_accessions)} RNA columns", flush=True)


if __name__ == "__main__":
    main()
