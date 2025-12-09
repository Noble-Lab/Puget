import sys
import os
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.append(repo_root)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from puget.end2end_model import PugetGenePredictor
from puget.end2end_data import get_seq_hicsubmat_dataloader
from puget.utils import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default=os.path.join(repo_root, "configs/inference/human_puget_k562.yaml"),
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at {args.config}")
    
    cfg = load_config(args.config)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Error: Configuration requests 'cuda', but no GPU was found."
        )
    
    DEVICE = torch.device(cfg.device)
    
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Running on device: {DEVICE}")
    
    pkl_df = pd.read_csv(cfg.pkl_list_file, header=None, names=["accession"])
    accession_list = pkl_df["accession"].to_list()
    TEST_HIC_WINDOWS_PKLS = [os.path.join(cfg.pkl_data_dir, f"{accession}.pkl") for accession in accession_list]
    
    test_seq_hicsubmat_loader = get_seq_hicsubmat_dataloader(
        bedpe_path=cfg.bedpe_path,
        windows_pkl_paths=TEST_HIC_WINDOWS_PKLS,
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

    all_preds = []

    with torch.inference_mode():
        for batch_idx, (seqs, imgs, total_counts, metas) in enumerate(tqdm(test_seq_hicsubmat_loader, desc="Embedding", unit="batch")):
            seqs = seqs.to(DEVICE, non_blocking=True)
            imgs = imgs.to(DEVICE, non_blocking=True)
            total_counts = total_counts.to(DEVICE, non_blocking=True)

            predictions = model(seqs, imgs, total_counts) # (B, 1)
            predictions = predictions.squeeze(-1).detach().cpu() #(B,)
            all_preds.append(predictions)

    preds_np = torch.cat(all_preds, dim=0).numpy()
    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    np.save(cfg.output_path, preds_np)

if __name__ == "__main__":
    main()
