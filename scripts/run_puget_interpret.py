import sys
import os
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.append(repo_root)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from tqdm.auto import tqdm
from captum.attr import InputXGradient

from puget.end2end_model import PugetGenePredictor
from puget.end2end_data import get_seq_hicsubmat_dataloader
from puget.utils import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default=os.path.join(repo_root, "configs/interpret/human_puget_saliency.yaml"),
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at {args.config}")
    
    cfg = load_config(args.config)
    
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configuration requests 'cuda', but no GPU was found.")
    
    DEVICE = torch.device(cfg.device)
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Using device: {DEVICE}")
    print(f"Batch size: {cfg.batch_size}")

    pkl_df = pd.read_csv(cfg.pkl_list_file, header=None, names=["accession"])
    accession_list = pkl_df["accession"].to_list()
    hic_windows_pkls = [os.path.join(cfg.pkl_data_dir, f"{acc}.pkl") for acc in accession_list]
    
    test_seq_hicsubmat_loader = get_seq_hicsubmat_dataloader(
        bedpe_path=cfg.bedpe_path,
        windows_pkl_paths=hic_windows_pkls,
        fasta_file=None,
        resolution=cfg.model.resolution,
        window_height=cfg.model.window_height,
        window_width=cfg.model.window_width,
        seq_length=cfg.model.seq_length,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,               
        transform_hic=None,
        seq_indices_npy=cfg.seq_indices_npy,              
    )

    model = PugetGenePredictor(
        enformer_ckpt_path = cfg.enformer_ckpt,
        hic_encoder_ckpt_path = cfg.hic_encoder_ckpt,
        hic_model_name = cfg.model.name,
        img_size_hw = (cfg.model.window_height, cfg.model.window_width),
        patch_size = cfg.model.patch_size,
        regressor_ckpt_path = cfg.regressor_ckpt,
    )
    model.to(DEVICE)
    model.eval()

    target_index = 0
    input_x_gradient = InputXGradient(model)

    os.makedirs(cfg.output_dir, exist_ok=True)
    seq_save_path = os.path.join(cfg.output_dir, cfg.seq_attr_filename)
    hic_save_path = os.path.join(cfg.output_dir, cfg.hic_attr_filename)

    seq_mmap = None
    hic_mmap = None
    start_idx = 0
    total_samples = len(test_seq_hicsubmat_loader.dataset)
    print(f"Total samples to process: {total_samples}")
    
    for i, (seqs, imgs, total_counts, metas) in tqdm(enumerate(test_seq_hicsubmat_loader), total=len(test_seq_hicsubmat_loader)):
        seqs = seqs.to(DEVICE, non_blocking=True)
        seqs.requires_grad_()
        imgs = imgs.to(DEVICE, non_blocking=True)
        imgs.requires_grad_()
        total_counts = total_counts.to(DEVICE, non_blocking=True)
        total_counts.requires_grad_(False)

        # Captum needs gradients enabled.
        with torch.enable_grad():
            attrs = input_x_gradient.attribute((seqs, imgs),
                                                target=target_index,
                                                additional_forward_args=total_counts)
            batch_seq_attr = attrs[0].detach().cpu().numpy()
            batch_hic_attr = attrs[1].detach().cpu().numpy()

        if seq_mmap is None:
            seq_shape = (total_samples,) + batch_seq_attr.shape[1:] 
            hic_shape = (total_samples,) + batch_hic_attr.shape[1:] 
            
            print(f"Initializing .npy files on disk.")
            print(f"Seq shape: {seq_shape}, HiC shape: {hic_shape}")

            # Create .npy files on disk immediately
            seq_mmap = open_memmap(seq_save_path, mode='w+', dtype='float32', shape=seq_shape)
            hic_mmap = open_memmap(hic_save_path, mode='w+', dtype='float32', shape=hic_shape)

        current_batch_size = batch_seq_attr.shape[0]
        end_idx = start_idx + current_batch_size
        
        seq_mmap[start_idx:end_idx] = batch_seq_attr
        hic_mmap[start_idx:end_idx] = batch_hic_attr
        
        start_idx = end_idx
        
    if seq_mmap is not None:
        seq_mmap.flush()
    if hic_mmap is not None:
        hic_mmap.flush()
    print(f"Done. Attributions saved incrementally to:\n{seq_save_path}\n{hic_save_path}")

if __name__ == "__main__":
    main()
