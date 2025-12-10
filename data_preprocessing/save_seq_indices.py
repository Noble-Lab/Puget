import os
import argparse
import numpy as np
import pandas as pd
import torch
from enformer_pytorch import GenomeIntervalDataset
from tqdm import tqdm

# ------------------------------------------------------------------
# The following function is adapted from the Enformer PyTorch repository.
# Source: https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/data.py
# License: MIT License
# ------------------------------------------------------------------
reverse_complement_map = torch.Tensor([3, 2, 1, 0, 4]).long()
def seq_indices_reverse_complement(seq_indices):
    complement = reverse_complement_map[seq_indices.long()]
    return torch.flip(complement, dims = (-1,))

def save_seq_indices(
    bed_filepath: str,
    fasta_filepath: str,
    output_npy: str,
    seq_length: int,
):
    with open(bed_filepath, "r") as f:
        ncols = len(f.readline().rstrip("\n").split("\t"))
    
    # print("ncols: ", ncols, flush=True)
    if ncols >= 6:
        cols = ["chrom","start","end","name","score","strand"]
    else:
        cols = ["chrom","start","end"]
    df = pd.read_csv(bed_filepath, sep="\t", header=None, names=cols)
    if "strand" not in df.columns:
        df["strand"] = "+"
    strand_list = df["strand"].to_list()
    
    ds = GenomeIntervalDataset(
        bed_file=bed_filepath,
        fasta_file=fasta_filepath,
        return_seq_indices=True,
        shift_augs=None,
        rc_aug=False,
        context_length=seq_length,
        return_augs=False,
    )
    
    output_array = np.zeros((len(ds), seq_length), dtype=np.uint8)
    
    for idx in tqdm(range(len(ds))):
        seq_indices = ds[idx]
        strand = strand_list[idx]
        if strand == "-":
            seq_indices = seq_indices_reverse_complement(seq_indices)
        
        output_array[idx,:] = seq_indices.to(torch.uint8).cpu().numpy()
    
    np.save(output_npy, output_array)

def main():
    parser = argparse.ArgumentParser(description="Generate stranded seq indices for regions.")
    parser.add_argument("--bed_filepath", type=str, required=True, help="Path to the BED file.")
    parser.add_argument("--fasta_filepath", type=str, required=True, help="Path to the FASTA file.")
    parser.add_argument(
        "--output_npy", type=str, required=True,
        help="Path to the output .npy file (shape: N x (tiles*center_bins) x D)."
    )
    parser.add_argument("--seq_length", type=int, default=192_000,
                        help="Model context length (bp) per example.")

    args = parser.parse_args()

    assert args.seq_length > 0, "--seq_length must be > 0"

    save_seq_indices(
        args.bed_filepath,
        args.fasta_filepath,
        args.output_npy,
        args.seq_length,
    )

if __name__ == "__main__":
    main()
