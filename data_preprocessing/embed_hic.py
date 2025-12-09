import sys
import os
import argparse
import gc

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.append(repo_root)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from tqdm.auto import tqdm

from puget.end2end_data import get_hicsubmat_dataloader
from puget.end2end_model import HiCEmbeddingGenerator

def run_inference_loop(model, accession_list, args, device):
    """Iterates through accessions, runs inference, and writes 5D tensor to disk."""
    mmap_file = None
    n_biosamples = len(accession_list)
    
    for bio_idx, accession in enumerate(tqdm(accession_list, desc="Biosamples")):
        
        pkl_path = os.path.join(args.pkl_data_dir, f"{accession}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Missing pickle file: {pkl_path}")

        loader = get_hicsubmat_dataloader(
            bedpe_path=args.bedpe_path,
            windows_pkl_paths=[pkl_path], 
            resolution=args.resolution,
            window_height=args.window_height,
            window_width=args.window_width,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            transform=None
        )

        total_regions = len(loader.dataset)
        region_start_idx = 0

        with torch.no_grad():
            for (imgs, total_counts, _) in loader:
                
                imgs = imgs.to(device, non_blocking=True)
                total_counts = total_counts.to(device, non_blocking=True)

                emb = model(imgs, total_counts)
                
                batch_emb = emb.cpu().numpy().astype(np.float16)
                batch_size = batch_emb.shape[0]

                if mmap_file is None:
                    shape = (n_biosamples, total_regions) + batch_emb.shape[1:]
                    print(f"Initializing .npy with shape: {shape}")
                    mmap_file = open_memmap(args.output_path, mode='w+', dtype='float16', shape=shape)

                end_idx = region_start_idx + batch_size
                mmap_file[bio_idx, region_start_idx:end_idx] = batch_emb
                
                region_start_idx = end_idx

        if mmap_file is not None:
            mmap_file.flush()

    print(f"Done! Saved Hi-C embeddings to: {args.output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Hi-C embeddings (N_Bio, M_Reg, R, C, D).")
    
    parser.add_argument("--pkl_list_file", type=str, required=True, help="Path to CSV list of accessions.")
    parser.add_argument("--pkl_data_dir", type=str, required=True, help="Directory containing .pkl files.")
    parser.add_argument("--bedpe_path", type=str, required=True, help="Path to BEDPE defining regions.")
    
    parser.add_argument("--hic_encoder_ckpt", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="vit_large_patch16")
    
    parser.add_argument("--output_path", type=str, required=True, help="Output .npy file path.")
    
    parser.add_argument("--window_height", type=int, default=192)
    parser.add_argument("--window_width", type=int, default=960)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=1000)
    
    parser.add_argument("--crop_rows", type=int, default=6, help="Target rows (e.g. 6).")
    parser.add_argument("--crop_cols", type=int, default=60, help="Target cols (e.g. 60).")
    
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda")
    
    gc.collect()
    torch.cuda.empty_cache()

    pkl_df = pd.read_csv(args.pkl_list_file, header=None, names=["accession"])
    accession_list = pkl_df["accession"].to_list()
    print(f"Found {len(accession_list)} biosamples to process.")

    print("Initializing HiCEmbeddingGenerator...")
    model = HiCEmbeddingGenerator(
        encoder_ckpt_path=args.hic_encoder_ckpt,
        model_name=args.model_name,
        img_size_hw=(args.window_height, args.window_width),
        patch_size=args.patch_size,
        crop_rows=args.crop_rows,
        crop_cols=args.crop_cols
    )
    model.to(device)
    model.eval()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    run_inference_loop(model, accession_list, args, device)

if __name__ == "__main__":
    main()
