import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate sequence x biosample expression label matrices.")

    parser.add_argument("--rna_dir", type=str, required=True,
                        help="Directory containing RSEM .tsv files.")
    parser.add_argument("--output_npy_dir", type=str, required=True,
                        help="Output directory for the final .npy label matrices.")
    
    parser.add_argument("--paired_table", type=str, default="data/accessions/supplementary-table-1.tsv",
                        help="Path to supplementary-table-1.tsv containing RNA-HiC mapping.")
    parser.add_argument("--gene_anno_file", type=str, default="data/gene_annotations/human_anno_tss_filtered.csv",
                        help="Path to human_anno_tss_filtered.csv annotation file.")
    parser.add_argument("--accession_dir", type=str, default="data/accessions",
                        help="Directory containing human_{split}_accessions.csv files.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    VAL_CHROM = ["chr4"]
    TEST_CHROM = ["chr5", "chr11", "chr14"]

    df_paired = pd.read_csv(args.paired_table, sep="\t")
    df_paired_human = df_paired[df_paired["Organism"] == "human"]
    rna2hic_accession = {}
    for _, row in df_paired_human.iterrows():
        rna2hic_accession[row["RNA Accession"]] = row["Hi-C Accession"]

    df_anno_filtered = pd.read_csv(args.gene_anno_file)
    gene_expr_subset = df_anno_filtered.copy()

    files_to_process = []
    for filename in os.listdir(args.rna_dir):
        if filename.endswith(".tsv"):
            rna_acc = filename.replace(".tsv", "")
            if rna_acc in rna2hic_accession:
                files_to_process.append((filename, rna2hic_accession[rna_acc]))

    for filename, hic_accession in tqdm(files_to_process, desc="Reading RSEM files"):
        input_path = os.path.join(args.rna_dir, filename)
        
        gene_expr_df = pd.read_csv(input_path, sep="\t", usecols=["gene_id", "TPM"])
        gene_expr_df["TPM"] = np.log2(gene_expr_df["TPM"] + 1)
        
        gene_expr_subset = pd.merge(
            left=gene_expr_subset,
            right=gene_expr_df, 
            how="inner",
            on="gene_id"
        )
        
        if gene_expr_subset["TPM"].isnull().any():
            raise ValueError(f"{hic_accession} TPM contains nan values")

        gene_expr_subset.rename(columns={"TPM": hic_accession}, inplace=True)

    os.makedirs(args.output_npy_dir, exist_ok=True)
    
    filtered_gene_expr_df = gene_expr_subset
    val_mask = filtered_gene_expr_df["chrom"].isin(VAL_CHROM)
    test_mask = filtered_gene_expr_df["chrom"].isin(TEST_CHROM)
    train_mask = ~(val_mask | test_mask)
    
    df_splits = {
        "train": filtered_gene_expr_df[train_mask],
        "val": filtered_gene_expr_df[val_mask],
        "test": filtered_gene_expr_df[test_mask]
    }
    
    num_seqs = {split: len(sub) for split, sub in df_splits.items()}

    acc_dict = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(args.accession_dir, f"human_{split}_accessions.csv")
        acc_dict[split] = pd.read_csv(path, names=["accession"], header=None, dtype=str)["accession"].to_list()
    
    num_bios = {split: len(lst) for split, lst in acc_dict.items()}
    
    row_idx = {
        split: {acc: i for i, acc in enumerate(lst)}
        for split, lst in acc_dict.items()
    }
    
    concat = {
        seq: {
            bio: np.zeros((num_bios[bio], num_seqs[seq]), dtype=np.float32)
            for bio in ["train", "val", "test"]
        }
        for seq in ["train", "val", "test"]
    }
    
    available_columns = set(filtered_gene_expr_df.columns)
    
    for bio_split in ["train", "val", "test"]:
        accessions = acc_dict[bio_split]
        
        for acc in tqdm(accessions, desc=f"Processing {bio_split} bio accessions"):
            if acc not in available_columns:
                print(f"Warning: Accession {acc} in {bio_split} list but not in expression data.")
                continue
            
            b_idx = row_idx[bio_split][acc]
            
            for seq_split in ["train", "val", "test"]:
                arr = df_splits[seq_split][acc].to_numpy(dtype=np.float32)
                concat[seq_split][bio_split][b_idx] = arr

    for seq in ["train", "val", "test"]:
        for bio in ["train", "val", "test"]:
            out_fn = f"label_{seq}seq_{bio}bio.npy"
            out_path = os.path.join(args.output_npy_dir, out_fn)
            np.save(out_path, concat[seq][bio])
            print(f"Saved {out_fn} with shape {concat[seq][bio].shape}")

if __name__ == "__main__":
    main()
