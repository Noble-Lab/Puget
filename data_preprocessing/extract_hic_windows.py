import argparse
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse import vstack, hstack
import pickle
from collections import defaultdict
from collections import Counter

def normalize_chrom_name(chrom):
    """Ensure chromosome names start with 'chr'."""
    return chrom if chrom.startswith("chr") else f"chr{chrom}"

def group_by_chrom(index):
    chrom_groups = defaultdict(list)
    for row in index:
        chrom_groups[row[0]].append(row[1:])
    return chrom_groups

def load_hic_data(hic_pkl_path):
    """Load and normalize Hi-C data from a pickle file."""
    with open(hic_pkl_path, 'rb') as f:
        data = pickle.load(f)
    normalized_data = {}
    for chrom, mat in data.items():
        key = normalize_chrom_name(chrom)
        normalized_data[key] = mat
    return normalized_data


def process_hic_data(data, window_height, window_width):
    """Process Hi-C data to make it symmetrical, adjust shapes, and return sparse matrices."""
    new_data = {}
    dataset_shape = {}
    total_count = np.float64(0.0)
    half_h = window_height // 2
    half_w = window_width // 2

    for chrom, hic in data.items():
        # Skip small chromosomes
        if hic.shape[0] < window_height or hic.shape[1] < window_width:
            # print(f"Skipping {chrom} as it is too small", flush=True)
            continue
        
        total_count += hic.data.sum(dtype=np.float64)

        # ---- symmetrise (vectorised) ----
        rows = hic.row.astype(np.int32)
        cols = hic.col.astype(np.int32)
        vals = hic.data.astype(np.float32)
        sym_rows = np.concatenate([rows, cols])
        sym_cols = np.concatenate([cols, rows])
        sym_vals  = np.concatenate([vals, vals])
        diag = sym_rows == sym_cols
        sym_vals[diag] *= 0.5
        
        # Build COO matrix
        sym_hic = coo_matrix((sym_vals, (sym_rows, sym_cols)), shape=hic.shape, dtype=np.float32).tocsr(copy=False)
        
        # Pad to at least window size
        new_shape = (max(hic.shape[0], window_height), max(hic.shape[1], window_width))
        if new_shape != sym_hic.shape:
            rows_pad = new_shape[0] - sym_hic.shape[0]
            cols_pad = new_shape[1] - sym_hic.shape[1]
            sym_hic = vstack(
                [hstack([sym_hic, coo_matrix((sym_hic.shape[0], cols_pad), dtype=np.float32)]),
                coo_matrix((rows_pad, new_shape[1]), dtype=np.float32)]
            ).tocsr()
        
        new_data[chrom] = sym_hic
        dataset_shape[chrom] = sym_hic.shape

    return new_data, dataset_shape, total_count

def load_bedpe(bedpe_path):
    """Load BEDPE with gene name and strand; require chr1 == chr2."""
    df = pd.read_csv(
        bedpe_path, sep="\t", header=None,
        names=["chr1","start1","end1","chr2","start2","end2","name","score","strand1","strand2"],
        dtype={"start1":"int64","end1":"int64","start2":"int64","end2":"int64"}
    )
    df["chr1"] = df["chr1"].apply(normalize_chrom_name)
    df["chr2"] = df["chr2"].apply(normalize_chrom_name)
    if not (df["chr1"] == df["chr2"]).all():
        raise ValueError("All entries in the BEDPE file must have chr1 equal to chr2.")
    return df

def generate_input_index(bedpe_df, dataset_shape, resolution):
    """Generate input indices from BEDPE entries."""
    idx_list = []
    for _, row in bedpe_df.iterrows():
        chrom = row['chr1']
        rs, re = row['start1'] // resolution, row['end1'] // resolution
        cs, ce = row['start2'] // resolution, row['end2'] // resolution
        if chrom not in dataset_shape:
            print(f"Skip input index chrom:{chrom},rs:{rs},cs:{cs} as chrom not in dataset_shape", flush=True)
            continue
        rsize, csize = dataset_shape[chrom]
        if rs >= rsize or cs >= csize:
            print(f"Skip input index chrom:{chrom},rs:{rs},cs:{cs} as rs/cs is too large", flush=True)
            continue
        re_clamped = min(rsize, re)
        ce_clamped = min(csize, ce)
        mid = (cs + ce_clamped) // 2
        
        strand = str(row.get("strand1", "+"))
        flip = (strand == "-")
        
        idx_list.append((chrom, rs, cs, re_clamped, ce_clamped, mid, flip))
    return idx_list

def save_inputs_to_pickle(input_index, data, window_height, window_width, 
                          output_pkl_path, output_invalid_bed_path, full_total_count):
    """Save each input window as a COO matrix in a pickle file."""
    invalid_entries = []
    input_dict = {}
    
    chrom_groups = group_by_chrom(input_index)
    for chrom, windows in chrom_groups.items():
        mat = data[chrom]
        for rs, cs, re, ce, mid, flip in windows:
            sub = mat[rs:rs + window_height, cs:cs + window_width].tocoo(copy=False)
            is_empty = (int(sub.nnz) == 0)
            if is_empty:
                invalid_entries.append((chrom, rs, cs))

            # indices (flip both axes if needed)
            r_idx = sub.row.astype(np.int32)
            c_idx = sub.col.astype(np.int32)
            if flip:
                r_idx = (window_height - 1) - r_idx
                c_idx = (window_width  - 1) - c_idx
            
            key = f"{chrom}:{rs},{cs}"
            input_dict[key] = {
                "row"  : r_idx.astype(np.int16),
                "col"  : c_idx.astype(np.int16),
                "data" : sub.data.astype(np.float16),
                "is_empty": bool(is_empty),
                "total_count": np.float64(full_total_count),
            }

    with open(output_pkl_path, "wb") as f:
        pickle.dump(input_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(input_dict)} windows to {output_pkl_path}")

    # Log invalid zero-nnz entry windows
    if len(invalid_entries) > 0:
        if output_invalid_bed_path:
            with open(output_invalid_bed_path, "w") as bed:
                for chrom, rs, cs in invalid_entries:
                    bed.write(f"{chrom}\t{rs}\t{cs}\n")
            print(f"Wrote {len(invalid_entries)} invalid entries to {output_invalid_bed_path}")
        else:
            print(f"WARNING: {len(invalid_entries)} invalid (empty) windows found.")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process Hi-C and BEDPE to generate sparse input windows"
    )
    parser.add_argument('--hic_pkl_path', type=str, required=True)
    parser.add_argument('--bedpe_path', type=str, required=True)
    parser.add_argument('--output_pkl_path', type=str, required=True)
    parser.add_argument('--output_invalid_bed_path', type=str, default=None)
    parser.add_argument('--resolution', type=int, default=1000)
    parser.add_argument('--window_height', type=int, default=192)
    parser.add_argument('--window_width', type=int, default=960)
    return parser.parse_args()

def main():
    args = parse_arguments()
    hic_data = load_hic_data(args.hic_pkl_path)
    processed_data, dataset_shape, total_reads = process_hic_data(
        hic_data, args.window_height, args.window_width
    )
    # print(f"Total reads in Hi-C data: {total_reads}")
    bedpe_df = load_bedpe(args.bedpe_path)
    input_index = generate_input_index(bedpe_df, dataset_shape, args.resolution)
    # print(f"Total number of input windows: {len(input_index)}")

    save_inputs_to_pickle(
        input_index, processed_data,
        args.window_height, args.window_width,
        args.output_pkl_path, args.output_invalid_bed_path,
        full_total_count=total_reads,
    )

if __name__ == "__main__":
    main()
