import sys
import os
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.append(repo_root)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
import gc
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from puget.end2end_model import EnformerGenePredictor
from puget.end2end_data import get_seq_indices_dataloader, seq_indices_to_one_hot
from puget.utils import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default=os.path.join(repo_root, "configs/inference/human_enformer.yaml"),
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found at {args.config}")
    cfg = load_config(args.config)
    
    if cfg.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Error: Configuration requests 'cuda', but no GPU was found.")
    
    DEVICE = torch.device(cfg.device)
    
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Running on device: {DEVICE}")
    
    dl = get_seq_indices_dataloader(
        seq_indices_npy=cfg.seq_indices_npy,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False
    )

    model = EnformerGenePredictor(
        enformer_ckpt_path=cfg.enformer_ckpt,
        regressor_ckpt_path=cfg.regressor_ckpt,
    )

    model.to(DEVICE)
    model.eval()

    all_preds = []

    with torch.inference_mode():
        for (seq_u8,) in tqdm(dl, desc="Embedding"):
            seq_long = seq_u8.to(DEVICE, non_blocking=True).long()
            seq_ohe = seq_indices_to_one_hot(seq_long)

            predictions = model(seq_ohe) # (B, 23)
            predictions = predictions.detach().cpu() #(B, 23)
            all_preds.append(predictions)

    preds_np = torch.cat(all_preds, dim=0).T.numpy()
    print(f"Saving predictions with shape: {preds_np.shape}")
    
    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    np.save(cfg.output_path, preds_np)

if __name__ == "__main__":
    main()
