import sys
import os
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.append(repo_root)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from puget.end2end_data import get_puget_perturbation_dataloader
from puget.end2end_model import PugetGenePredictor
from puget.utils import load_config

def main():
    parser = argparse.ArgumentParser(
        description="Run Puget on paired (original/perturbed) Hi-C windows."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=os.path.join(repo_root, "configs/interpretation/k562_perturbation.yaml"),
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at {args.config}")
    
    cfg = load_config(args.config)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configuration requests 'cuda', but no GPU was found.")
    device = torch.device(cfg.device)
    
    gc.collect()
    torch.cuda.empty_cache()

    if not hasattr(cfg, 'k_median_npy') or cfg.k_median_npy is None:
        raise ValueError("Config must specify 'k_median_npy' for perturbation.")
        
    distance_band_median = np.load(cfg.k_median_npy)
    if distance_band_median.shape != (576,):
        raise ValueError(f"k_median_npy must have shape (576,), got {distance_band_median.shape}")

    dataloader = get_puget_perturbation_dataloader(
        windows_pkl_path=cfg.windows_pkl,
        targets=cfg.targets_csv,
        seq_indices_npy=cfg.seq_indices_npy,
        k_median=distance_band_median,
        H=cfg.model.window_height,
        W=cfg.model.window_width,
        fill_mode="k_expected",
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        transform_hic=None,
    )

    puget_model = PugetGenePredictor(
        enformer_ckpt_path=cfg.enformer_ckpt,
        hic_encoder_ckpt_path=cfg.hic_encoder_ckpt,
        hic_model_name=cfg.model.name,
        img_size_hw=(cfg.model.window_height, cfg.model.window_width),
        patch_size=cfg.model.patch_size,
        regressor_ckpt_path=cfg.regressor_ckpt,
    )
    puget_model.to(device)
    puget_model.eval()

    all_predictions_original = []
    all_predictions_perturbed = []

    with torch.inference_mode():
        for (seq_onehot_batch,
             img_original_batch,
             img_perturbed_batch,
             total_counts_batch,
             metas_batch) in tqdm(dataloader, desc="Inference", unit="batch"):

            seq_onehot_batch = seq_onehot_batch.to(device, non_blocking=True)
            img_original_batch = img_original_batch.to(device, non_blocking=True)
            img_perturbed_batch = img_perturbed_batch.to(device, non_blocking=True)
            total_counts_batch = total_counts_batch.to(device, non_blocking=True)

            preds_orig = puget_model(seq_onehot_batch, img_original_batch, total_counts_batch)
            preds_orig = preds_orig.squeeze(-1).detach().cpu()  # (B,)
            all_predictions_original.append(preds_orig)

            preds_pert = puget_model(seq_onehot_batch, img_perturbed_batch, total_counts_batch)
            preds_pert = preds_pert.squeeze(-1).detach().cpu()
            all_predictions_perturbed.append(preds_pert)
            
    y_orig = torch.cat(all_predictions_original, dim=0).numpy()
    y_pert = torch.cat(all_predictions_perturbed, dim=0).numpy()
    delta = y_pert - y_orig

    targets_df = pd.read_csv(cfg.targets_csv)
    if len(targets_df) != len(y_orig):
        raise RuntimeError(f"Row mismatch: targets={len(targets_df)} vs preds={len(y_orig)}")

    targets_df = targets_df.copy()
    targets_df["y_orig"] = y_orig
    targets_df["y_pert"] = y_pert
    targets_df["delta"] = delta

    os.makedirs(os.path.dirname(cfg.out_csv), exist_ok=True)
    targets_df.to_csv(cfg.out_csv, index=False)
    print(f"Perturbation results saved to: {cfg.out_csv}")

if __name__ == "__main__":
    main()
