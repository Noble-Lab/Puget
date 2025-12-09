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
from numpy.lib.format import open_memmap
from tqdm.auto import tqdm

from puget.end2end_data import get_seq_indices_dataloader, seq_indices_to_one_hot
from puget.end2end_model import SequenceEmbeddingGenerator

def run_inference_loop(model, loader, output_path, device):
    mmap_file = None
    start_idx = 0
    total_samples = len(loader.dataset)
    
    print(f"Processing {total_samples} samples...")

    with torch.no_grad():
        for (seq_u8,) in tqdm(loader, desc="Generating Embeddings"):
            seq_long = seq_u8.to(device, non_blocking=True).long()
            seq_ohe = seq_indices_to_one_hot(seq_long)

            embeddings = model(seq_ohe)
            embeddings = embeddings.flatten(start_dim=1)
            batch_emb = embeddings.cpu().numpy().astype(np.float16)

            if mmap_file is None:
                shape = (total_samples,) + batch_emb.shape[1:]
                mmap_file = open_memmap(output_path, mode='w+', dtype='float16', shape=shape)

            batch_size = batch_emb.shape[0]
            end_idx = start_idx + batch_size
            mmap_file[start_idx:end_idx] = batch_emb
            
            start_idx = end_idx

    if mmap_file is not None:
        mmap_file.flush()
    
    print(f"Done! Embeddings saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Enformer embeddings from sequence indices.")
    
    parser.add_argument("--seq_indices_npy", type=str, required=True, 
                        help="Path to the .npy file containing (N, L) uint8 sequence indices.")
    parser.add_argument("--enformer_ckpt", type=str, required=True, 
                        help="Path to the pre-trained Enformer checkpoint.")
    
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Output path for the .npy embeddings file.")
    
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--crop_seq_bins", type=int, default=750, 
                        help="Number of bins to crop from the center (Default: 750).")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    device = torch.device("cuda")
    
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading Sequence Indices...")
    loader = get_seq_indices_dataloader(
        seq_indices_npy=args.seq_indices_npy,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False 
    )

    print(f"Initializing SequenceEmbeddingGenerator...")
    model = SequenceEmbeddingGenerator(
        enformer_ckpt_path=args.enformer_ckpt, 
        crop_seq_bins=args.crop_seq_bins
    )
    model.to(device)
    model.eval()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    run_inference_loop(model, loader, args.output_path, device)

if __name__ == "__main__":
    main()
